from pathlib import Path

import pytest

from app.domain import Decision, InspectionRecord
from app.procedure_routing import POLICY_SECTION_ROUTES, routes_for_record
from app.rag import SopRetriever

DOCS = Path(__file__).resolve().parents[1] / "docs"


@pytest.mark.parametrize(
    ("reason", "document_id", "parent_section"),
    [
        ("SCORE_AT_OR_ABOVE_REJECT_THRESHOLD", "SOP-QA-04", "Immediate actions"),
        ("SCORE_IN_REVIEW_BAND", "SOP-REV-02", "Required review steps"),
        ("IMAGE_TOO_BLURRY", "SOP-REV-02", "Required review steps"),
        ("ARTIFACT_NOT_VERIFIED", "SOP-ESC-03", "Escalate when"),
    ],
)
def test_policy_reason_routes_to_approved_parent_section(reason, document_id, parent_section):
    record = InspectionRecord(
        inspection_id="test",
        part_id="part",
        image_uri="image.png",
        decision=Decision.INSPECT,
        metadata={"critical_policy": {"reason_codes": [reason]}},
    )

    assert routes_for_record(record)[0] == POLICY_SECTION_ROUTES[reason][0]
    assert routes_for_record(record)[0].document_id == document_id
    assert routes_for_record(record)[0].parent_section == parent_section


def test_local_retriever_prioritizes_exact_policy_routed_section():
    record = InspectionRecord(
        inspection_id="test",
        part_id="part",
        image_uri="image.png",
        decision=Decision.INSPECT,
        metadata={"critical_policy": {"reason_codes": ["SCORE_IN_REVIEW_BAND"]}},
    )

    reference = SopRetriever(DOCS).retrieve(record, top_k=1)[0]

    assert (reference.document_id, reference.section) == ("SOP-REV-02", "Required review steps")
