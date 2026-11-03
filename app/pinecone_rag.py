"""Pinecone-backed semantic retrieval for approved inspection procedures.

Pinecone's integrated embedding index turns the ``text`` field into vectors on
write and embeds a text query on read.  The policy is still the authority over
which documents are eligible for a decision; Pinecone only ranks sections in
that approved subset.
"""
from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

from .domain import InspectionRecord, ProcedureReference
from .rag import REASON_DOCUMENTS, ProcedureChunk, load_procedure_chunks


class PineconeIndex(Protocol):
    def search_records(self, *, namespace: str, query: dict[str, Any], fields: list[str]) -> Any: ...

    def upsert_records(self, namespace: str, records: list[dict[str, str]]) -> Any: ...


def chunk_id(chunk: ProcedureChunk) -> str:
    """Stable identity means a re-index overwrites the same SOP section."""
    content = f"{chunk.document_id}\n{chunk.section}\n{chunk.text}".encode()
    return f"{chunk.document_id}:{hashlib.sha256(content).hexdigest()[:16]}"


def records_for_chunks(chunks: list[ProcedureChunk]) -> list[dict[str, str]]:
    """Build the source-of-truth records expected by Pinecone's field map."""
    return [
        {
            "_id": chunk_id(chunk),
            "text": chunk.embedding_text,
            "document_id": chunk.document_id,
            "title": chunk.title,
            "section": chunk.section,
            "excerpt": " ".join(chunk.text.split())[:500],
        }
        for chunk in chunks
    ]


def index_approved_procedures(index: PineconeIndex, docs_dir: Path, namespace: str) -> int:
    records = records_for_chunks(load_procedure_chunks(docs_dir))
    if not records:
        raise ValueError(f"No approved SOP chunks found in {docs_dir}")
    index.upsert_records(namespace=namespace, records=records)
    return len(records)


def _preferred_document_ids(record: InspectionRecord) -> list[str]:
    reasons = record.metadata.get("critical_policy", {}).get("reason_codes", [])
    return list(dict.fromkeys(doc for reason in reasons for doc in REASON_DOCUMENTS.get(reason, ())))


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

    def __init__(self, index: PineconeIndex, namespace: str) -> None:
        self.index = index
        self.namespace = namespace

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
        return cls(client.Index(host=index_host), os.getenv("PINECONE_NAMESPACE", "grid-inspection-sop-v1"))

    def retrieve(self, record: InspectionRecord, top_k: int = 2) -> list[ProcedureReference]:
        allowed_document_ids = _preferred_document_ids(record)
        document_filter = (
            {"document_id": {"$in": allowed_document_ids}} if allowed_document_ids else None
        )
        return self._search(_policy_query(record), top_k, document_filter)

    def search(self, query: str, top_k: int = 3) -> list[ProcedureReference]:
        """Semantic free-text retrieval, used by the versioned RAG benchmark."""
        return self._search(query, top_k, None)

    def _search(
        self, query_text: str, top_k: int, metadata_filter: dict[str, Any] | None
    ) -> list[ProcedureReference]:
        query: dict[str, Any] = {"inputs": {"text": query_text}, "top_k": top_k}
        if metadata_filter:
            query["filter"] = metadata_filter
        response = self.index.search_records(
            namespace=self.namespace,
            query=query,
            fields=["document_id", "title", "section", "excerpt"],
        )
        result = _value(response, "result", response)
        hits = _value(result, "hits", []) or []
        references: list[ProcedureReference] = []
        for hit in hits:
            fields = _value(hit, "fields", {}) or {}
            document_id = _value(fields, "document_id")
            title = _value(fields, "title")
            section = _value(fields, "section")
            excerpt = _value(fields, "excerpt")
            score = _value(hit, "_score", _value(hit, "score", 0.0))
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
