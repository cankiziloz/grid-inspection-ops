from __future__ import annotations

import os
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Literal
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .domain import CaptureMetadata, Decision, ReviewReasonCode, ReviewResolution
from .explain import OllamaProcedureExplainer, SafeTemplateExplainer
from .graph import build_inspection_graph
from .model_artifacts import download_grid_model
from .operations_briefing import build_operations_briefing_snapshot
from .pinecone_rag import PineconeSopRetriever
from .rag import SopRetriever
from .repository import SqliteInspectionRepository
from .vision import GridPatchCoreVision

load_dotenv(Path(__file__).parent.parent / ".env")

app = FastAPI(title="Grid Inspection Ops", version="0.1.0")
runtime_root = Path(os.getenv("RUNTIME_DIR", "runtime"))
runtime_root.mkdir(parents=True, exist_ok=True)
app.mount("/files", StaticFiles(directory=runtime_root), name="runtime-files")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv(
        "CORS_ORIGINS", "http://localhost:8080,http://127.0.0.1:8080"
    ).split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class ReviewRequest(BaseModel):
    reviewer_id: str = Field(min_length=1, max_length=128)
    final_decision: Decision
    reason_code: ReviewReasonCode
    reason: str = Field(min_length=3, max_length=2000)


def runtime():
    if not hasattr(app.state, "graph"):
        repo_id = os.environ.get("GRID_MODEL_REPO_ID")
        if not repo_id:
            raise RuntimeError("GRID_MODEL_REPO_ID must identify the published Hugging Face model")
        artifact = download_grid_model(
            repo_id=repo_id,
            revision=os.getenv("GRID_MODEL_REVISION", "main"),
            cache_dir=os.getenv("GRID_MODEL_CACHE_DIR", "runtime/models/grid-patchcore-v1"),
        )
        repository = SqliteInspectionRepository(os.getenv("DATABASE_PATH", "runtime/inspections.db"))
        vision = GridPatchCoreVision(str(artifact.model_dir), str(artifact.stn_path), model_version=f"{artifact.model_id}@{artifact.revision}", artifact_verified=True)
        app.state.repository = repository
        retrieval_provider = os.getenv("RETRIEVAL_PROVIDER", "local").lower()
        if retrieval_provider == "pinecone":
            retriever = PineconeSopRetriever.from_environment()
        elif retrieval_provider == "local":
            retriever = SopRetriever(Path(__file__).parent.parent / "docs")
        else:
            raise RuntimeError("RETRIEVAL_PROVIDER must be 'local' or 'pinecone'")
        if os.getenv("EXPLANATION_PROVIDER", "template").lower() == "ollama":
            explainer = OllamaProcedureExplainer(
                model=os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b"),
                base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
            )
        else:
            explainer = SafeTemplateExplainer()
        app.state.graph = build_inspection_graph(vision, repository, explainer, retriever)
    return app.state.graph, app.state.repository


@app.get("/healthz")
def healthz():
    return {"status": "ok", "model_repo": os.getenv("GRID_MODEL_REPO_ID")}


async def _run_inspection(
    file: UploadFile,
    part_id: str,
    capture: CaptureMetadata | None = None,
    response: Response | None = None,
):
    if file.content_type not in {"image/jpeg", "image/png"}:
        raise HTTPException(415, "Only PNG and JPEG grid images are accepted")
    payload = await file.read()
    if len(payload) > 15 * 1024 * 1024:
        raise HTTPException(413, "Image exceeds the 15 MB limit")
    if capture and sha256(payload).hexdigest() != capture.image_sha256:
        raise HTTPException(422, "image_sha256 does not match the uploaded image")
    graph, repository = runtime()
    if capture:
        existing = repository.get_by_frame_id(capture.frame_id)
        if existing:
            if response:
                response.headers["X-Idempotent-Replay"] = "true"
            return existing
    suffix = Path(file.filename or "image.png").suffix.lower() or ".png"
    upload_dir = Path("runtime/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    image_path = upload_dir / f"{uuid4()}{suffix}"
    image_path.write_bytes(payload)
    try:
        state = {"part_id": part_id, "image_path": str(image_path)}
        if capture:
            state["capture"] = capture.model_dump()
        result = graph.invoke(state)
        return result["record"]
    except Exception as error:
        image_path.unlink(missing_ok=True)
        raise HTTPException(503, f"Inspection unavailable: {error}") from error


@app.post("/v1/inspections")
async def create_inspection(file: UploadFile = File(...), part_id: str = Form(...)):
    """Manual demo harness; production ingress should use /v1/edge/inspections."""
    return await _run_inspection(file, part_id)


@app.post("/v1/edge/inspections")
async def create_edge_inspection(
    response: Response,
    file: UploadFile = File(...),
    part_id: str = Form(...),
    station_id: str = Form(...),
    camera_id: str = Form(...),
    frame_id: str = Form(...),
    captured_at: str = Form(...),
    image_sha256: str = Form(...),
):
    """Camera/edge ingress with immutable capture metadata and frame idempotency."""
    try:
        datetime.fromisoformat(captured_at.replace("Z", "+00:00"))
    except ValueError as error:
        raise HTTPException(422, "captured_at must be an ISO-8601 timestamp") from error
    capture = CaptureMetadata(
        station_id=station_id,
        camera_id=camera_id,
        frame_id=frame_id,
        captured_at=captured_at,
        image_sha256=image_sha256.lower(),
    )
    return await _run_inspection(file, part_id, capture, response)


@app.get("/v1/inspections")
def list_inspections(limit: int = 50):
    _, repository = runtime()
    return repository.list_recent(limit=min(max(limit, 1), 200))


@app.get("/v1/inspections/{inspection_id}")
def get_inspection(inspection_id: str):
    _, repository = runtime()
    record = repository.get(inspection_id)
    if not record:
        raise HTTPException(404, "Inspection not found")
    return record


@app.post("/v1/inspections/{inspection_id}/review")
def resolve_review(inspection_id: str, request: ReviewRequest):
    if request.final_decision not in {Decision.PASS, Decision.REJECT}:
        raise HTTPException(422, "Human review must resolve to PASS or REJECT")
    _, repository = runtime()
    review = ReviewResolution(
        reviewer_id=request.reviewer_id,
        final_decision=request.final_decision,
        reason_code=request.reason_code,
        reason=request.reason,
        reviewed_at=datetime.now(timezone.utc).isoformat(),
    )
    try:
        return repository.resolve_review(inspection_id, review)
    except KeyError as error:
        raise HTTPException(404, "Inspection not found") from error
    except ValueError as error:
        raise HTTPException(409, str(error)) from error


@app.get("/v1/dashboard")
def dashboard():
    _, repository = runtime()
    return repository.dashboard_summary()


@app.get("/v1/operations/briefing-snapshot")
def operations_briefing_snapshot(
    period: Literal["24h", "7d", "30d", "all"] = "24h",
):
    """LLM-free, typed evidence package for a future operations briefing."""
    _, repository = runtime()
    return build_operations_briefing_snapshot(repository.dashboard_summary(), period)
