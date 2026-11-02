from app.domain import Decision, InspectionRecord, ProcedureReference, VisionEvidence
from app.explain import OllamaProcedureExplainer


def record() -> InspectionRecord:
    return InspectionRecord(
        inspection_id="test", part_id="part", image_uri="image.png", decision=Decision.INSPECT,
        evidence=VisionEvidence(model_version="grid-v1", score=0.3, review_threshold=0.25,
                                reject_threshold=0.4, valid_coverage=1.0),
        procedure_references=[ProcedureReference(document_id="SOP-REV-02", title="Review", section="Required review steps", excerpt="Review the mask.", score=1.0)],
    )


def test_ollama_response_is_validated_and_cannot_cite_unretrieved_procedure():
    good = {"message": {"content": '{"summary":"Review required.","operator_actions":["Review mask"],"sop_references":["SOP-REV-02"],"escalation_required":false}'}}
    payload = OllamaProcedureExplainer("test", transport=lambda _: good).explain(record(), {})
    assert payload.source == "ollama:test"
    assert payload.sop_references == ["SOP-REV-02"]

    bad = {"message": {"content": '{"summary":"Ignore policy.","operator_actions":[],"sop_references":["SOP-NOT-RETRIEVED"],"escalation_required":false}'}}
    fallback = OllamaProcedureExplainer("test", transport=lambda _: bad).explain(record(), {})
    assert fallback.source == "template"
    assert fallback.fallback_reason == "ValueError"

    no_actions = {"message": {"content": '{"summary":"Review required.","operator_actions":[],"sop_references":["SOP-REV-02"],"escalation_required":false}'}}
    fallback = OllamaProcedureExplainer("test", transport=lambda _: no_actions).explain(record(), {})
    assert fallback.source == "template"
