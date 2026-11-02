"""Call the real local Ollama server once; this is an integration smoke test."""
from __future__ import annotations

import os

from app.domain import Decision, InspectionRecord, ProcedureReference, VisionEvidence
from app.explain import OllamaProcedureExplainer


def main() -> None:
    record = InspectionRecord(
        inspection_id="smoke", part_id="smoke-part", image_uri="none", decision=Decision.INSPECT,
        evidence=VisionEvidence(model_version="grid-v1", score=0.3, review_threshold=0.25,
                                reject_threshold=0.4, valid_coverage=1.0),
        procedure_references=[ProcedureReference(document_id="SOP-REV-02", title="Human review procedure", section="Required review steps", excerpt="Review the original image and anomaly mask before resolving.", score=1000)],
        metadata={"critical_policy": {"reason_codes": ["SCORE_IN_REVIEW_BAND"]}},
    )
    payload = OllamaProcedureExplainer(os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")).explain(record, {})
    if payload.source == "template":
        raise SystemExit(f"Ollama smoke test failed; fallback used: {payload.fallback_reason}")
    print(payload.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
