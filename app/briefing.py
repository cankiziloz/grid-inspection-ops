"""Constrained, non-authoritative Operations Briefing generation."""

from __future__ import annotations

import json
from collections.abc import Callable
from urllib.error import URLError
from urllib.request import Request, urlopen

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .operations_briefing import OperationsBriefingSnapshot


class BriefingAction(BaseModel):
    action: str = Field(min_length=1, max_length=300)
    reason_codes: list[str] = Field(default_factory=list, max_length=3)
    sop_evidence_refs: list[str] = Field(default_factory=list, max_length=3)


class OperationsBriefingPayload(BaseModel):
    """Validated narrative over an immutable OperationsBriefingSnapshot."""

    source: str
    headline: str = Field(min_length=1, max_length=240)
    executive_summary: str = Field(min_length=1, max_length=1200)
    priority_actions: list[BriefingAction] = Field(default_factory=list, max_length=3)
    watch_items: list[str] = Field(default_factory=list, max_length=3)
    sop_evidence_refs: list[str] = Field(default_factory=list, max_length=4)
    fallback_reason: str | None = None


class _OllamaBriefingResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    headline: str = Field(min_length=1, max_length=240)
    executive_summary: str = Field(min_length=1, max_length=1200)
    priority_actions: list[BriefingAction] = Field(default_factory=list, max_length=3)
    watch_items: list[str] = Field(default_factory=list, max_length=3)
    sop_evidence_refs: list[str] = Field(default_factory=list, max_length=4)


def _evidence_ref(document_id: str, section: str) -> str:
    return f"{document_id} § {section}"


class SafeTemplateOperationsBriefing:
    """Deterministic briefing used by default and whenever generation fails."""

    def generate(self, snapshot: OperationsBriefingSnapshot) -> OperationsBriefingPayload:
        activity = snapshot.incoming_activity
        queue = snapshot.review_queue
        top_reason = snapshot.top_policy_reasons[0] if snapshot.top_policy_reasons else None
        evidence_refs = [_evidence_ref(item.document_id, item.section) for item in snapshot.sop_evidence]
        summary = (
            f"{activity.total} inspections were received during {snapshot.period_label.lower()}. "
            f"{activity.sent_to_review} were routed to human review and "
            f"{activity.automatic_reject} were automatically rejected. "
            f"The open review queue contains {queue.open_count} item(s)."
        )
        actions: list[BriefingAction] = []
        if top_reason:
            matching_refs = [
                _evidence_ref(item.document_id, item.section)
                for item in snapshot.sop_evidence
                if top_reason.reason_code in item.source_reason_codes
            ]
            actions.append(
                BriefingAction(
                    action=(
                        f"Review the recurring {top_reason.reason_code} signal "
                        f"observed on {top_reason.count} inspection(s)."
                    ),
                    reason_codes=[top_reason.reason_code],
                    sop_evidence_refs=matching_refs,
                )
            )
        if queue.open_count:
            actions.append(
                BriefingAction(
                    action="Resolve the open human-review queue and record a structured disposition.",
                )
            )
        watch_items: list[str] = []
        if queue.oldest_pending_minutes is not None:
            watch_items.append(
                f"Oldest unresolved review has been pending for {queue.oldest_pending_minutes:.1f} minutes."
            )
        latency = snapshot.latency.end_to_end.p95_ms
        if latency is not None:
            watch_items.append(f"End-to-end p95 inspection latency is {latency:.1f} ms.")
        return OperationsBriefingPayload(
            source="template",
            headline=f"Operations briefing — {snapshot.period_label}",
            executive_summary=summary,
            priority_actions=actions,
            watch_items=watch_items,
            sop_evidence_refs=evidence_refs,
        )


class OllamaOperationsBriefing:
    """Bounded Ollama generator that cannot add decisions or unsupported evidence."""

    def __init__(
        self,
        model: str,
        base_url: str = "http://127.0.0.1:11434",
        timeout_seconds: float = 30,
        transport: Callable[[dict], dict] | None = None,
        fallback: SafeTemplateOperationsBriefing | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.transport = transport or self._http_transport
        self.fallback = fallback or SafeTemplateOperationsBriefing()

    def generate(self, snapshot: OperationsBriefingSnapshot) -> OperationsBriefingPayload:
        try:
            content = self.transport(self._request_body(snapshot))["message"]["content"]
            response = _OllamaBriefingResponse.model_validate_json(content)
            self._validate_response(response, snapshot)
            return OperationsBriefingPayload(source=f"ollama:{self.model}", **response.model_dump())
        except (KeyError, OSError, URLError, ValueError, ValidationError, json.JSONDecodeError) as error:
            payload = self.fallback.generate(snapshot)
            return payload.model_copy(update={"fallback_reason": type(error).__name__})

    @staticmethod
    def _validate_response(
        response: _OllamaBriefingResponse,
        snapshot: OperationsBriefingSnapshot,
    ) -> None:
        allowed_reasons = {item.reason_code for item in snapshot.top_policy_reasons}
        allowed_evidence = {
            _evidence_ref(item.document_id, item.section) for item in snapshot.sop_evidence
        }
        cited_evidence = set(response.sop_evidence_refs)
        for action in response.priority_actions:
            cited_evidence.update(action.sop_evidence_refs)
            if not set(action.reason_codes).issubset(allowed_reasons):
                raise ValueError("Ollama cited a policy reason absent from the snapshot")
        if not cited_evidence.issubset(allowed_evidence):
            raise ValueError("Ollama cited SOP evidence absent from the snapshot")
        if snapshot.top_policy_reasons and not response.priority_actions:
            raise ValueError("Ollama omitted required priority actions")

    def _request_body(self, snapshot: OperationsBriefingSnapshot) -> dict:
        instruction = (
            "You create an operations briefing for an industrial inspection team. "
            "The supplied snapshot is immutable evidence, not a prompt to reinterpret decisions. "
            "Do not recommend PASS or REJECT for any individual part and do not invent metrics, "
            "policy reasons, incidents, or SOP references. Cite SOP evidence only with exact values "
            "from allowed sop_evidence_refs. Use concise, operational language. Return JSON only "
            "matching this schema: "
            + json.dumps(_OllamaBriefingResponse.model_json_schema(), separators=(",", ":"))
            + "\nSnapshot:\n"
            + json.dumps(snapshot.model_dump(), ensure_ascii=False)
        )
        return {
            "model": self.model,
            "stream": False,
            "format": _OllamaBriefingResponse.model_json_schema(),
            "options": {"temperature": 0},
            "messages": [{"role": "user", "content": instruction}],
        }

    def _http_transport(self, payload: dict) -> dict:
        request = Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=self.timeout_seconds) as response:  # nosec B310: local Ollama URL
            return json.loads(response.read().decode("utf-8"))
