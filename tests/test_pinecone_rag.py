from pathlib import Path

from app.domain import Decision, InspectionRecord
from app.pinecone_rag import PineconeSopRetriever, index_approved_procedures, records_for_chunks
from app.rag import load_procedure_chunks

DOCS = Path(__file__).resolve().parents[1] / "docs"


class FakePineconeIndex:
    def __init__(self):
        self.upserts: list[tuple[str, list[dict[str, str]]]] = []
        self.searches: list[dict] = []
        self.documents = FakePineconeDocuments(self)


class FakePineconeDocuments:
    def __init__(self, index: FakePineconeIndex):
        self.index = index

    def upsert(self, *, namespace, documents):
        self.index.upserts.append((namespace, documents))

    def search(self, *, namespace, score_by, top_k, include_fields, filter=None):
        self.index.searches.append(
            {
                "namespace": namespace,
                "score_by": score_by,
                "top_k": top_k,
                "include_fields": include_fields,
                "filter": filter,
            }
        )
        return {
            "matches": [
                {
                    "score": 0.91,
                    "document_id": "SOP-QA-04",
                    "title": "Reject and Quarantine Policy",
                    "section": "Immediate actions",
                    "excerpt": "Quarantine the part and preserve evidence.",
                }
            ]
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
    assert all(
        {"chunk_text", "document_id", "title", "section", "parent_section", "chunk_index", "excerpt"}
        <= record.keys()
        for record in records
    )


def test_embedding_field_is_configurable_for_a_different_pinecone_schema():
    record = records_for_chunks(load_procedure_chunks(DOCS), embedding_field="body")[0]

    assert "body" in record
    assert "chunk_text" not in record


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
    search = index.searches[0]
    assert search["filter"] == {
        "$or": [
            {
                "document_id": {"$eq": "SOP-QA-04"},
                "parent_section": {"$eq": "Immediate actions"},
            }
        ]
    }
    assert search["score_by"][0]["type"] == "embed"
    assert search["score_by"][0]["field"] == "chunk_text"
    assert "REJECT" in search["score_by"][0]["query"]


def test_free_text_semantic_search_has_no_policy_filter():
    index = FakePineconeIndex()
    PineconeSopRetriever(index, "grid-inspection-sop-v1").search("What happens after a reject?")

    assert index.searches[0]["filter"] is None
