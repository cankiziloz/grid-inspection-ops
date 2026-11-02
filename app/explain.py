from __future__ import annotations

import json
from typing import Callable
from urllib.error import URLError
from urllib.request import Request, urlopen

from pydantic import BaseModel, Field, ValidationError

from .domain import ExplanationPayload, InspectionRecord


class _OllamaResponse(BaseModel):
    summary: str = Field(min_length=1, max_length=1200)
    operator_actions: list[str] = Field(default_factory=list, max_length=5)
    sop_references: list[str] = Field(default_factory=list, max_length=3)
    escalation_required: bool = False


class SafeTemplateExplainer:
    """Deterministic fallback until a local Ollama service is configured."""

    def explain(self, record: InspectionRecord, history: dict) -> ExplanationPayload:
        evidence = record.evidence
        assert evidence is not None
        references = ", ".join(
            f"{item.document_id} § {item.section}" for item in record.procedure_references
        ) or "No procedure reference available"
        return ExplanationPayload(
            source="template",
            summary=(
            f"Grid policy returned {record.decision}: score {evidence.score:.4f}; "
            f"review threshold {evidence.review_threshold:.4f}; "
            f"reject threshold {evidence.reject_threshold:.4f}; "
            f"valid coverage {evidence.valid_coverage:.1%}. "
            f"Relevant procedures: {references}."
            ),
            operator_actions=record.checklist,
            sop_references=[item.document_id for item in record.procedure_references],
        )


class OllamaProcedureExplainer:
    """Constrained local generator; it may explain but cannot alter a decision."""

    def __init__(
        self,
        model: str,
        base_url: str = "http://127.0.0.1:11434",
        timeout_seconds: float = 30,
        transport: Callable[[dict], dict] | None = None,
        fallback: SafeTemplateExplainer | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.transport = transport or self._http_transport
        self.fallback = fallback or SafeTemplateExplainer()

    def explain(self, record: InspectionRecord, history: dict) -> ExplanationPayload:
        allowed_references = {item.document_id for item in record.procedure_references}
        try:
            response = _OllamaResponse.model_validate_json(self.transport(self._request_body(record, history))["message"]["content"])
            if not set(response.sop_references).issubset(allowed_references):
                raise ValueError("Ollama cited a procedure that was not retrieved")
            if record.decision.value in {"INSPECT", "REJECT"} and not response.operator_actions:
                raise ValueError("Ollama omitted required operator actions")
            return ExplanationPayload(source=f"ollama:{self.model}", **response.model_dump())
        except (KeyError, OSError, URLError, ValueError, ValidationError, json.JSONDecodeError) as error:
            payload = self.fallback.explain(record, history)
            return payload.model_copy(update={"fallback_reason": type(error).__name__})

    def _request_body(self, record: InspectionRecord, history: dict) -> dict:
        evidence = record.evidence
        assert evidence is not None
        procedures = [item.model_dump(include={"document_id", "section", "excerpt"}) for item in record.procedure_references]
        context = {
            "initial_decision": record.decision.value,
            "final_decision": record.final_decision.value if record.final_decision else None,
            "policy_reason_codes": record.metadata.get("critical_policy", {}).get("reason_codes", []),
            "score": evidence.score,
            "review_threshold": evidence.review_threshold,
            "reject_threshold": evidence.reject_threshold,
            "valid_coverage": evidence.valid_coverage,
            "recent_history": history,
            "retrieved_procedures": procedures,
        }
        instruction = (
            "You explain an industrial inspection record. The initial decision is immutable: never recommend changing it. "
            "For INSPECT or REJECT, operator_actions must contain at least one concrete next action; never state that no action is required. "
            "Use only SOP IDs supplied in retrieved_procedures. Return JSON only, matching this schema: "
            + json.dumps(_OllamaResponse.model_json_schema(), separators=(",", ":"))
            + "\nRecord context:\n"
            + json.dumps(context, ensure_ascii=False)
        )
        return {"model": self.model, "stream": False, "format": _OllamaResponse.model_json_schema(), "options": {"temperature": 0}, "messages": [{"role": "user", "content": instruction}]}

    def _http_transport(self, payload: dict) -> dict:
        request = Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=self.timeout_seconds) as response:  # nosec B310: local configurable Ollama endpoint
            return json.loads(response.read().decode("utf-8"))
