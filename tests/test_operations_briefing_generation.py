import json

from test_operations_briefing import dashboard_summary

from app.briefing import OllamaOperationsBriefing, SafeTemplateOperationsBriefing
from app.operations_briefing import build_operations_briefing_snapshot


def snapshot(tmp_path):
    (tmp_path / "review.md").write_text(
        "# Human Review Procedure\n\n"
        "**Document ID:** SOP-REV-02\n\n"
        "## Required review steps\n\n"
        "Confirm image quality, review the evidence, and record a final disposition.\n",
        encoding="utf-8",
    )
    (tmp_path / "ingestion.md").write_text(
        "# Image Ingestion Procedure\n\n"
        "**Document ID:** SOP-ING-01\n\n"
        "## Image acceptance rules\n\n"
        "Route frames that fail capture acceptance checks.\n",
        encoding="utf-8",
    )
    return build_operations_briefing_snapshot(dashboard_summary(), "24h", docs_dir=tmp_path)


def response(content: dict) -> dict:
    return {"message": {"content": json.dumps(content)}}


def test_ollama_briefing_accepts_only_snapshot_reason_and_sop_evidence(tmp_path):
    payload = {
        "headline": "Coverage review required",
        "executive_summary": "Human review is elevated during the selected period.",
        "priority_actions": [
            {
                "action": "Review the capture-quality signal.",
                "reason_codes": ["LOW_VALID_COVERAGE"],
                "sop_evidence_refs": ["SOP-REV-02 § Required review steps"],
            }
        ],
        "watch_items": ["Review queue remains open."],
        "sop_evidence_refs": ["SOP-ING-01 § Image acceptance rules"],
    }

    briefing = OllamaOperationsBriefing("test", transport=lambda _: response(payload)).generate(
        snapshot(tmp_path)
    )

    assert briefing.source == "ollama:test"
    assert briefing.priority_actions[0].reason_codes == ["LOW_VALID_COVERAGE"]


def test_ollama_briefing_falls_back_for_unapproved_evidence_or_decision_field(tmp_path):
    bad_evidence = {
        "headline": "Unsupported",
        "executive_summary": "Unsupported evidence was selected.",
        "priority_actions": [
            {
                "action": "Do something.",
                "reason_codes": ["LOW_VALID_COVERAGE"],
                "sop_evidence_refs": ["SOP-UNKNOWN § Invented section"],
            }
        ],
        "watch_items": [],
        "sop_evidence_refs": [],
    }
    fallback = OllamaOperationsBriefing("test", transport=lambda _: response(bad_evidence)).generate(
        snapshot(tmp_path)
    )
    assert fallback.source == "template"
    assert fallback.fallback_reason == "ValueError"

    forbidden_decision = {
        **bad_evidence,
        "decision": "REJECT",
        "priority_actions": [
            {
                "action": "Review capture quality.",
                "reason_codes": ["LOW_VALID_COVERAGE"],
                "sop_evidence_refs": ["SOP-REV-02 § Required review steps"],
            }
        ],
    }
    fallback = OllamaOperationsBriefing(
        "test", transport=lambda _: response(forbidden_decision)
    ).generate(snapshot(tmp_path))
    assert fallback.source == "template"
    assert fallback.fallback_reason == "ValidationError"


def test_template_briefing_uses_snapshot_without_an_llm(tmp_path):
    briefing = SafeTemplateOperationsBriefing().generate(snapshot(tmp_path))

    assert briefing.source == "template"
    assert briefing.sop_evidence_refs
    assert briefing.priority_actions[0].reason_codes == ["LOW_VALID_COVERAGE"]
