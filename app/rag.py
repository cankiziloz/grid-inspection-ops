"""Deterministic retrieval over the approved portfolio SOP documents.

This is intentionally not an agent: policy reason codes select a constrained
set of procedures, then lexical scoring ranks sections inside that set. It
provides stable, testable context for a later LLM explanation layer.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .domain import InspectionRecord, ProcedureReference

_TOKEN = re.compile(r"[a-z0-9][a-z0-9_-]{1,}")

REASON_DOCUMENTS: dict[str, tuple[str, ...]] = {
    "SCORE_AT_OR_ABOVE_REJECT_THRESHOLD": ("SOP-QA-04", "SOP-ESC-03"),
    "SCORE_IN_REVIEW_BAND": ("SOP-REV-02", "SOP-ESC-03"),
    "LOW_VALID_COVERAGE": ("SOP-REV-02", "SOP-ESC-03"),
    "MISSING_IMAGE_QUALITY": ("SOP-REV-02",),
    "IMAGE_RESOLUTION_TOO_LOW": ("SOP-REV-02", "SOP-ING-01"),
    "IMAGE_EXPOSURE_OUT_OF_RANGE": ("SOP-REV-02", "SOP-ING-01"),
    "IMAGE_CONTRAST_TOO_LOW": ("SOP-REV-02", "SOP-ING-01"),
    "IMAGE_TOO_BLURRY": ("SOP-REV-02", "SOP-ING-01"),
    "ARTIFACT_NOT_VERIFIED": ("SOP-ESC-03",),
    "NON_FINITE_ANOMALY_SCORE": ("SOP-ESC-03",),
    "INVALID_THRESHOLD_ORDER": ("SOP-ESC-03",),
    "ALL_CRITICAL_GATES_PASSED": ("SOP-ING-01",),
}


@dataclass(frozen=True)
class ProcedureChunk:
    document_id: str
    title: str
    section: str
    text: str

    @property
    def tokens(self) -> set[str]:
        return set(_TOKEN.findall(self.text.lower()))


class SopRetriever:
    def __init__(self, docs_dir: Path) -> None:
        self.chunks = self._load(docs_dir)
        if not self.chunks:
            raise ValueError(f"No approved SOP chunks found in {docs_dir}")

    @staticmethod
    def _load(docs_dir: Path) -> list[ProcedureChunk]:
        chunks: list[ProcedureChunk] = []
        for path in sorted(docs_dir.glob("*.md")):
            if path.name == "README.md":
                continue
            text = path.read_text(encoding="utf-8")
            document_id_match = re.search(r"\*\*Document ID:\*\*\s*([^\s]+)", text)
            title_match = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
            if not document_id_match or not title_match:
                continue
            document_id, title = document_id_match.group(1), title_match.group(1).strip()
            sections = re.split(r"^##\s+", text, flags=re.MULTILINE)
            for raw in sections[1:]:
                lines = raw.strip().splitlines()
                if len(lines) < 2:
                    continue
                section, body = lines[0].strip(), "\n".join(lines[1:]).strip()
                if body:
                    chunks.append(ProcedureChunk(document_id, title, section, body))
        return chunks

    def retrieve(self, record: InspectionRecord, top_k: int = 2) -> list[ProcedureReference]:
        reasons = record.metadata.get("critical_policy", {}).get("reason_codes", [])
        preferred_order = list(
            dict.fromkeys(doc for reason in reasons for doc in REASON_DOCUMENTS.get(reason, ()))
        )
        preferred_rank = {document_id: rank for rank, document_id in enumerate(preferred_order)}
        query = " ".join([record.decision.value, *reasons, *(record.checklist or [])]).lower()
        query_tokens = set(_TOKEN.findall(query))
        ranked: list[tuple[float, ProcedureChunk]] = []
        for chunk in self.chunks:
            overlap = len(query_tokens & chunk.tokens)
            # A policy-code mapping is a hard governance rule; lexical overlap
            # only chooses the most relevant section within that procedure.
            # The first mapped document is the governed primary procedure.
            # Lower-priority supporting procedures can never outrank it merely
            # because they happen to share more words with the query.
            policy_boost = 1000.0 - (100.0 * preferred_rank[chunk.document_id]) if chunk.document_id in preferred_rank else 0.0
            section_bonus = 0.1 * len(query_tokens & set(_TOKEN.findall(chunk.section.lower())))
            ranked.append((policy_boost + overlap + section_bonus, chunk))
        selected = sorted(ranked, key=lambda item: (-item[0], item[1].document_id, item[1].section))[:top_k]
        return [
            ProcedureReference(
                document_id=chunk.document_id,
                title=chunk.title,
                section=chunk.section,
                excerpt=" ".join(chunk.text.split())[:500],
                score=round(score, 3),
            )
            for score, chunk in selected
        ]
