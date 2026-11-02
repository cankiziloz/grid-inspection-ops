from __future__ import annotations

import os
from pathlib import Path
from datetime import datetime, timezone
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field
from fastapi.staticfiles import StaticFiles

from .domain import Decision, ReviewResolution
from fastapi.middleware.cors import CORSMiddleware

from .explain import SafeTemplateExplainer
from .graph import build_inspection_graph
from .model_artifacts import download_grid_model
from .rag import SopRetriever
from .repository import SqliteInspectionRepository
from .vision import GridPatchCoreVision

app = FastAPI(title="Grid Inspection Ops", version="0.1.0")
runtime_root = Path(os.getenv("RUNTIME_DIR", "runtime"))
runtime_root.mkdir(parents=True, exist_ok=True)
app.mount("/files", StaticFiles(directory=runtime_root), name="runtime-files")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:8080").split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class ReviewRequest(BaseModel):
    reviewer_id: str = Field(min_length=1, max_length=128)
    final_decision: Decision
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
        retriever = SopRetriever(Path(__file__).parent.parent / "docs")
        app.state.graph = build_inspection_graph(vision, repository, SafeTemplateExplainer(), retriever)
    return app.state.graph, app.state.repository


@app.get("/healthz")
def healthz():
    return {"status": "ok", "model_repo": os.getenv("GRID_MODEL_REPO_ID")}


@app.post("/v1/inspections")
async def create_inspection(file: UploadFile = File(...), part_id: str = Form(...)):
    if file.content_type not in {"image/jpeg", "image/png"}:
        raise HTTPException(415, "Only PNG and JPEG grid images are accepted")
    payload = await file.read()
    if len(payload) > 15 * 1024 * 1024:
        raise HTTPException(413, "Image exceeds the 15 MB limit")
    suffix = Path(file.filename or "image.png").suffix.lower() or ".png"
    upload_dir = Path("runtime/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    image_path = upload_dir / f"{uuid4()}{suffix}"
    image_path.write_bytes(payload)
    try:
        graph, _ = runtime()
        result = graph.invoke({"part_id": part_id, "image_path": str(image_path)})
        return result["record"]
    except Exception as error:
        image_path.unlink(missing_ok=True)
        raise HTTPException(503, f"Inspection unavailable: {error}") from error


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
    return repository.recent_summary()
