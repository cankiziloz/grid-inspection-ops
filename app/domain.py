from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Decision(str, Enum):
    PASS = "PASS"
    INSPECT = "INSPECT"
    REJECT = "REJECT"
    MODEL_ERROR = "MODEL_ERROR"


class ReviewStatus(str, Enum):
    NOT_REQUIRED = "NOT_REQUIRED"
    PENDING = "PENDING"
    RESOLVED = "RESOLVED"


class ReviewResolution(BaseModel):
    reviewer_id: str = Field(min_length=1, max_length=128)
    final_decision: Decision
    reason: str = Field(min_length=3, max_length=2000)
    reviewed_at: str


class DefectRegion(BaseModel):
    x: int
    y: int
    width: int
    height: int
    area_px: int
    peak_score: float


class ImageQuality(BaseModel):
    """Measured input properties used by the fail-closed policy gate."""

    model_config = ConfigDict(frozen=True)

    width: int = Field(gt=0)
    height: int = Field(gt=0)
    mean_luminance: float = Field(ge=0, le=1)
    contrast: float = Field(ge=0)
    laplacian_variance: float = Field(ge=0)
    acceptable: bool
    failure_reasons: list[str] = Field(default_factory=list)


class VisionEvidence(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_version: str
    score: float
    review_threshold: float
    reject_threshold: float
    valid_coverage: float = Field(ge=0, le=1)
    artifact_verified: bool = False
    quality: ImageQuality | None = None
    mask_uri: str | None = None
    regions: list[DefectRegion] = Field(default_factory=list)


class InspectionRecord(BaseModel):
    inspection_id: str
    part_id: str
    image_uri: str
    decision: Decision
    final_decision: Decision | None = None
    review_status: ReviewStatus = ReviewStatus.NOT_REQUIRED
    review: ReviewResolution | None = None
    evidence: VisionEvidence | None = None
    explanation: str | None = None
    checklist: list[str] = Field(default_factory=list)
    review_note: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
