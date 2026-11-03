"""Verify live Pinecone retrieval constrained by inspection policy section routes."""
from __future__ import annotations

import json
from pathlib import Path

from dotenv import load_dotenv

from app.domain import Decision, InspectionRecord
from app.pinecone_rag import PineconeSopRetriever


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env")
    retriever = PineconeSopRetriever.from_environment()
    cases = json.loads((root / "tests" / "fixtures" / "policy_routing_cases.json").read_text())
    passed = 0

    for index, case in enumerate(cases, start=1):
        record = InspectionRecord(
            inspection_id=f"routing-benchmark-{index}",
            part_id="routing-benchmark",
            image_uri="not-used.png",
            decision=Decision(case["decision"]),
            metadata={"critical_policy": {"reason_codes": [case["reason_code"]]}},
        )
        results = retriever.retrieve(record, top_k=3)
        expected = (case["document_id"], case["parent_section"])
        returned = [(item.document_id, item.section) for item in results]
        success = bool(returned) and returned[0] == expected
        passed += success
        print(
            f"reason={case['reason_code']}\n"
            f"  expected={expected[0]} § {expected[1]}\n"
            f"  retrieved={[f'{document_id} § {section}' for document_id, section in returned] or ['NO_RESULT']}\n"
            f"  result={'PASS' if success else 'FAIL'}"
        )

    total = len(cases)
    accuracy = passed / total if total else 0.0
    print(f"\npolicy-routed section accuracy@1: {passed}/{total} = {accuracy:.1%}")
    if passed != total:
        raise SystemExit("Policy routing benchmark failed")


if __name__ == "__main__":
    main()
