from __future__ import annotations

from .domain import InspectionRecord


class SafeTemplateExplainer:
    """Deterministic fallback until a local Ollama service is configured."""

    def explain(self, record: InspectionRecord, history: dict) -> str:
        evidence = record.evidence
        assert evidence is not None
        return (
            f"Grid policy returned {record.decision}: score {evidence.score:.4f}; "
            f"review threshold {evidence.review_threshold:.4f}; "
            f"reject threshold {evidence.reject_threshold:.4f}; "
            f"valid coverage {evidence.valid_coverage:.1%}."
        )
