from pathlib import Path

from app.domain import Decision, InspectionRecord
from app.pinecone_rag import PineconeSopRetriever, index_approved_procedures
from app.rag import load_procedure_chunks

DOCS = Path(__file__).resolve().parents[1] / "docs"


class FakePineconeIndex:
    def __init__(self):
        self.upserts: list[tuple[str, list[dict[str, str]]]] = []
        self.searches: list[dict] = []

    def upsert_records(self, *, namespace, records):
        self.upserts.append((namespace, records))

    def search_records(self, *, namespace, query, fields):
        self.searches.append({"namespace": namespace, "query": query, "fields": fields})
        return {
            "result": {
                "hits": [
                    {
                        "_score": 0.91,
                        "fields": {
                            "document_id": "SOP-QA-04",
                            "title": "Reject and Quarantine Policy",
                            "section": "Immediate actions",
                            "excerpt": "Quarantine the part and preserve evidence.",
                        },
                    }
                ]
            }
        }


def test_sync_writes_one_stable_record_per_approved_section():
    index = FakePineconeIndex()
    count = index_approved_procedures(index, DOCS, "grid-inspection-sop-v1")

    expected_chunks = load_procedure_chunks(DOCS)
    assert count == len(expected_chunks)
    namespace, records = index.upserts[0]
    assert namespace == "grid-inspection-sop-v1"
    assert len(records) == len(expected_chunks)
    assert all(record["_id"].startswith("SOP-") for record in records)
    assert all({"text", "document_id", "title", "section", "excerpt"} <= record.keys() for record in records)


def test_policy_retrieval_uses_pinecone_semantic_search_inside_allowed_sops():
    index = FakePineconeIndex()
    retriever = PineconeSopRetriever(index, "grid-inspection-sop-v1")
    record = InspectionRecord(
        inspection_id="test",
        part_id="part",
        image_uri="image.png",
        decision=Decision.REJECT,
        checklist=["Quarantine the part."],
        metadata={"critical_policy": {"reason_codes": ["SCORE_AT_OR_ABOVE_REJECT_THRESHOLD"]}},
    )

    references = retriever.retrieve(record)

    assert references[0].document_id == "SOP-QA-04"
    query = index.searches[0]["query"]
    assert query["filter"] == {"document_id": {"$in": ["SOP-QA-04", "SOP-ESC-03"]}}
    assert "REJECT" in query["inputs"]["text"]


def test_free_text_semantic_search_has_no_policy_filter():
    index = FakePineconeIndex()
    PineconeSopRetriever(index, "grid-inspection-sop-v1").search("What happens after a reject?")

    assert "filter" not in index.searches[0]["query"]
