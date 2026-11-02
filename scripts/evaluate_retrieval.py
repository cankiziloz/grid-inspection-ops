"""Run the small, versioned retrieval regression set used by the test suite."""
from __future__ import annotations

from pathlib import Path

from app.domain import Decision, InspectionRecord
from app.rag import SopRetriever


CASES = {
    "SCORE_AT_OR_ABOVE_REJECT_THRESHOLD": "SOP-QA-04",
    "SCORE_IN_REVIEW_BAND": "SOP-REV-02",
    "IMAGE_TOO_BLURRY": "SOP-REV-02",
    "ARTIFACT_NOT_VERIFIED": "SOP-ESC-03",
}


def main() -> None:
    retriever = SopRetriever(Path(__file__).resolve().parents[1] / "docs")
    correct = 0
    for reason, expected in CASES.items():
        record = InspectionRecord(inspection_id=reason, part_id="evaluation", image_uri="none", decision=Decision.INSPECT,
                                  metadata={"critical_policy": {"reason_codes": [reason]}})
        actual = retriever.retrieve(record, top_k=1)[0].document_id
        correct += actual == expected
        print(f"{reason}: expected={expected}, retrieved={actual}")
    print(f"Recall@1: {correct / len(CASES):.1%} ({correct}/{len(CASES)})")


if __name__ == "__main__":
    main()
