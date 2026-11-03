"""Versioned retrieval benchmark metrics for approved SOP documents."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class SearchRetriever(Protocol):
    def search(self, query: str, top_k: int = 3): ...


@dataclass(frozen=True)
class RetrievalCase:
    query: str
    document_id: str
    section: str


def load_cases(path: Path) -> list[RetrievalCase]:
    return [RetrievalCase(**item) for item in json.loads(path.read_text())]


def evaluate(retriever: SearchRetriever, cases: list[RetrievalCase]) -> dict[str, float]:
    if not cases:
        raise ValueError("Retrieval benchmark has no cases")
    recall_at_1 = recall_at_3 = section_at_1 = 0
    reciprocal_ranks: list[float] = []
    for case in cases:
        results = retriever.search(case.query, top_k=3)
        document_positions = [index for index, item in enumerate(results, start=1) if item.document_id == case.document_id]
        recall_at_1 += bool(document_positions and document_positions[0] == 1)
        recall_at_3 += bool(document_positions)
        section_at_1 += bool(results and (results[0].document_id, results[0].section) == (case.document_id, case.section))
        reciprocal_ranks.append(1 / document_positions[0] if document_positions else 0.0)
    total = len(cases)
    return {"recall_at_1": recall_at_1 / total, "recall_at_3": recall_at_3 / total, "mrr": sum(reciprocal_ranks) / total, "section_accuracy_at_1": section_at_1 / total}
