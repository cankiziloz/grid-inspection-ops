"""Pinecone-backed semantic retrieval for approved inspection procedures."""
from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

from .domain import InspectionRecord, ProcedureReference
from .procedure_routing import documents_for_reasons, routes_for_record
from .rag import ProcedureChunk, load_procedure_chunks


class PineconeDocuments(Protocol):
    def upsert(self, *, namespace: str, documents: list[dict[str, Any]]) -> Any: ...

    def search(
        self,
        *,
        namespace: str,
        score_by: list[dict[str, str]],
        top_k: int,
        include_fields: list[str],
        filter: dict[str, Any] | None = None,
    ) -> Any: ...


class PineconeIndex(Protocol):
    documents: PineconeDocuments


def chunk_id(chunk: ProcedureChunk) -> str:
    """Stable identity means a re-index overwrites the same SOP section."""
    content = f"{chunk.document_id}\n{chunk.section}\n{chunk.text}".encode()
    return f"{chunk.document_id}:{hashlib.sha256(content).hexdigest()[:16]}"


def records_for_chunks(
    chunks: list[ProcedureChunk], embedding_field: str = "chunk_text"
) -> list[dict[str, Any]]:
    """Build documents that satisfy the index's configured embedding field."""
    return [
        {
            "_id": chunk_id(chunk),
            embedding_field: chunk.embedding_text,
            "document_id": chunk.document_id,
            "title": chunk.title,
            "section": chunk.section,
            "parent_section": chunk.section,
            "chunk_index": 0,
            "excerpt": " ".join(chunk.text.split())[:500],
        }
        for chunk in chunks
    ]


def index_approved_procedures(
    index: PineconeIndex,
    docs_dir: Path,
    namespace: str,
    embedding_field: str = "chunk_text",
) -> int:
    records = records_for_chunks(load_procedure_chunks(docs_dir), embedding_field)
    if not records:
        raise ValueError(f"No approved SOP chunks found in {docs_dir}")
    index.documents.upsert(namespace=namespace, documents=records)
    return len(records)


def _policy_query(record: InspectionRecord) -> str:
    reasons = record.metadata.get("critical_policy", {}).get("reason_codes", [])
    return " ".join(
        [
            "Grid inspection operating procedure.",
            f"Decision: {record.decision.value}.",
            "Critical reasons: " + ", ".join(reasons) + ".",
            "Required actions: " + " ".join(record.checklist),
        ]
    )


def _value(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, Mapping):
        return item.get(key, default)
    return getattr(item, key, default)


class PineconeSopRetriever:
    """Semantic retriever for an already-provisioned integrated-embedding index."""

    def __init__(self, index: PineconeIndex, namespace: str, embedding_field: str = "chunk_text") -> None:
        self.index = index
        self.namespace = namespace
        self.embedding_field = embedding_field

    @classmethod
    def from_environment(cls) -> PineconeSopRetriever:
        api_key = os.getenv("PINECONE_API_KEY")
        index_host = os.getenv("PINECONE_INDEX_HOST")
        if not api_key or not index_host:
            raise RuntimeError(
                "PINECONE_API_KEY and PINECONE_INDEX_HOST are required when RETRIEVAL_PROVIDER=pinecone"
            )
        try:
            from pinecone import Pinecone
        except ImportError as error:
            raise RuntimeError("Install the Pinecone extra: pip install -e '.[rag]'") from error
        client = Pinecone(api_key=api_key)
        return cls(
            client.index(host=index_host),
            os.getenv("PINECONE_NAMESPACE", "grid-inspection-sop-v1"),
            os.getenv("PINECONE_EMBEDDING_FIELD", "chunk_text"),
        )

    def retrieve(self, record: InspectionRecord, top_k: int = 2) -> list[ProcedureReference]:
        routes = routes_for_record(record)
        if routes:
            routing_filter: dict[str, Any] = {
                "$or": [
                    {
                        "document_id": {"$eq": route.document_id},
                        "parent_section": {"$eq": route.parent_section},
                    }
                    for route in routes
                ]
            }
        else:
            reasons = record.metadata.get("critical_policy", {}).get("reason_codes", [])
            allowed_document_ids = documents_for_reasons(reasons)
            routing_filter = {"document_id": {"$in": list(allowed_document_ids)}} if allowed_document_ids else None
        return self._search(_policy_query(record), top_k, routing_filter)

    def search(self, query: str, top_k: int = 3) -> list[ProcedureReference]:
        """Semantic free-text retrieval, used by the versioned RAG benchmark."""
        return self._search(query, top_k, None)

    def _search(
        self, query_text: str, top_k: int, metadata_filter: dict[str, Any] | None
    ) -> list[ProcedureReference]:
        response = self.index.documents.search(
            namespace=self.namespace,
            score_by=[{"type": "embed", "field": self.embedding_field, "query": query_text}],
            top_k=top_k,
            include_fields=["document_id", "title", "section", "parent_section", "excerpt"],
            filter=metadata_filter,
        )
        hits = _value(response, "matches", []) or []
        references: list[ProcedureReference] = []
        for hit in hits:
            document_id = _value(hit, "document_id")
            title = _value(hit, "title")
            section = _value(hit, "section")
            excerpt = _value(hit, "excerpt")
            score = _value(hit, "score", 0.0)
            if all(isinstance(value, str) and value for value in (document_id, title, section, excerpt)):
                references.append(
                    ProcedureReference(
                        document_id=document_id,
                        title=title,
                        section=section,
                        excerpt=excerpt,
                        score=round(float(score), 6),
                    )
                )
        return references
