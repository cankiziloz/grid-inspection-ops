from pathlib import Path

import pytest

from app.domain import Decision, InspectionRecord
from app.rag import SopRetriever

DOCS = Path(__file__).resolve().parents[1] / "docs"


@pytest.mark.parametrize(
    ("reason", "expected_document"),
    [
        ("SCORE_AT_OR_ABOVE_REJECT_THRESHOLD", "SOP-QA-04"),
        ("SCORE_IN_REVIEW_BAND", "SOP-REV-02"),
        ("IMAGE_TOO_BLURRY", "SOP-REV-02"),
        ("ARTIFACT_NOT_VERIFIED", "SOP-ESC-03"),
    ],
)
def test_policy_reason_retrieves_expected_procedure(reason, expected_document):
    record = InspectionRecord(
        inspection_id="test", part_id="part", image_uri="image.png", decision=Decision.INSPECT,
        metadata={"critical_policy": {"reason_codes": [reason]}},
    )
    references = SopRetriever(DOCS).retrieve(record)
    assert references[0].document_id == expected_document
    assert references[0].excerpt


def test_retriever_only_uses_approved_sop_files():
    references = SopRetriever(DOCS).retrieve(
        InspectionRecord(inspection_id="test", part_id="part", image_uri="image.png", decision=Decision.PASS,
                         metadata={"critical_policy": {"reason_codes": ["ALL_CRITICAL_GATES_PASSED"]}})
    )
    assert all(reference.document_id.startswith("SOP-") for reference in references)
