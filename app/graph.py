from __future__ import annotations

from pathlib import Path
from typing import Annotated, TypedDict
from uuid import uuid4

from langgraph.graph import END, START, StateGraph

from .domain import Decision, InspectionRecord, ReviewStatus
from .policy import checklist_for, evaluate
from .ports import ExplanationService, GridVisionModel, InspectionRepository
from .rag import SopRetriever


class InspectionState(TypedDict, total=False):
    inspection_id: str
    part_id: str
    image_path: str
    record: InspectionRecord
    history: dict


def build_inspection_graph(
    vision: GridVisionModel,
    repository: InspectionRepository,
    explainer: ExplanationService,
    retriever: SopRetriever | None = None,
):
    def validate_input(state: InspectionState) -> InspectionState:
        path = Path(state["image_path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        return {"inspection_id": state.get("inspection_id", str(uuid4()))}

    def run_grid_model(state: InspectionState) -> InspectionState:
        evidence = vision.inspect(Path(state["image_path"]))
        outcome = evaluate(evidence)
        record = InspectionRecord(
            inspection_id=state["inspection_id"],
            part_id=state["part_id"],
            image_uri=state["image_path"],
            decision=outcome.decision,
            final_decision=None if outcome.decision is Decision.INSPECT else outcome.decision,
            review_status=ReviewStatus.PENDING if outcome.decision is Decision.INSPECT else ReviewStatus.NOT_REQUIRED,
            evidence=evidence,
            checklist=checklist_for(outcome.decision, evidence),
            metadata={"critical_policy": {"version": outcome.policy_version, "reason_codes": list(outcome.reason_codes)}},
        )
        return {"record": record}

    def load_history(state: InspectionState) -> InspectionState:
        return {"history": repository.recent_summary()}

    def add_explanation(state: InspectionState) -> InspectionState:
        record = state["record"]
        # The explanation is post-decision only; it cannot mutate record.decision.
        payload = explainer.explain(record, state["history"])
        return {"record": record.model_copy(update={"explanation": payload.summary, "explanation_payload": payload})}

    def retrieve_procedures(state: InspectionState) -> InspectionState:
        if retriever is None:
            return {}
        record = state["record"]
        return {"record": record.model_copy(update={"procedure_references": retriever.retrieve(record)})}

    def persist(state: InspectionState) -> InspectionState:
        repository.save(state["record"])
        return {}

    graph = StateGraph(InspectionState)
    graph.add_node("validate_input", validate_input)
    graph.add_node("run_grid_model", run_grid_model)
    graph.add_node("load_history", load_history)
    graph.add_node("retrieve_procedures", retrieve_procedures)
    graph.add_node("add_explanation", add_explanation)
    graph.add_node("persist", persist)
    graph.add_edge(START, "validate_input")
    graph.add_edge("validate_input", "run_grid_model")
    graph.add_edge("run_grid_model", "load_history")
    graph.add_edge("load_history", "retrieve_procedures")
    graph.add_edge("retrieve_procedures", "add_explanation")
    graph.add_edge("add_explanation", "persist")
    graph.add_edge("persist", END)
    return graph.compile()
