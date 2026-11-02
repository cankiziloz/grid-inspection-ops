from __future__ import annotations

from pathlib import Path
from typing import Protocol

from .domain import ExplanationPayload, InspectionRecord, ReviewResolution, VisionEvidence


class GridVisionModel(Protocol):
    def inspect(self, image_path: Path) -> VisionEvidence: ...


class InspectionRepository(Protocol):
    def save(self, record: InspectionRecord) -> None: ...

    def recent_summary(self, limit: int = 50) -> dict: ...

    def resolve_review(self, inspection_id: str, review: ReviewResolution) -> InspectionRecord: ...


class ExplanationService(Protocol):
    def explain(self, record: InspectionRecord, history: dict) -> ExplanationPayload: ...
