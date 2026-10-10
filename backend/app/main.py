"""FastAPI application: routes and worker lifecycle.

Owner: C. One server worker owns the camera, so run with a single uvicorn
worker and without autoreload while testing camera stability.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import cv2
from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .actions import Dispatcher
from .camera import CaptureWorker, build_source, crop_region
from .compiler import build_preview_text, compile_instruction
from .config import Settings, load_settings
from .db import Database
from .evidence import EvidenceStore, encode_jpeg
from .history import SUPPORTED_EXAMPLES, answer_question
from .pipeline import InferenceWorker
from .runtime import WorkflowRuntime
from .schemas import (
    ActionStatus,
    CameraHealth,
    ROI,
    SourceKind,
    Workflow,
)
from .vision import build_provider

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


# --------------------------------------------------------------------------
# Request models
# --------------------------------------------------------------------------


class CompileRequest(BaseModel):
    instruction: str = Field(min_length=1, max_length=2000)
    camera_id: str = "cam_01"
    roi: Optional[ROI] = None


class SaveWorkflowRequest(BaseModel):
    workflow: Workflow


class PatchWorkflowRequest(BaseModel):
    """Any provided field replaces the current one and bumps the version."""

    target_description: Optional[str] = None
    roi: Optional[ROI] = None
    duration_seconds: Optional[float] = None
    min_samples: Optional[int] = None
    max_gap_seconds: Optional[float] = None
    clear_samples: Optional[int] = None
    enabled: Optional[bool] = None
    preview_validated: Optional[bool] = None


class HistoryQueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    workflow_id: Optional[str] = None


# --------------------------------------------------------------------------
# Application state
# --------------------------------------------------------------------------


class AppState:
    """Everything with a lifetime longer than a request."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.db = Database(settings.database_path)
        self.evidence = EvidenceStore(settings.evidence_dir)
        self.dispatcher = Dispatcher(
            self.db, settings.allowlist, timeout=settings.webhook_timeout_seconds
        )
        self.provider = build_provider(settings)
        self.capture = CaptureWorker(
            build_source(settings),
            fps=settings.preview_fps,
            stale_after_seconds=settings.stale_after_seconds,
            on_frame=self._fan_out_frame,
        )
        self.runtimes: dict[str, WorkflowRuntime] = {}
        self.workers: dict[str, InferenceWorker] = {}
        self._concurrency = asyncio.Semaphore(2)

    # -- frames ---------------------------------------------------------

    def _fan_out_frame(self, frame) -> None:
        """Push every captured frame into each runtime's evidence buffer."""
        if not self.runtimes:
            return
        buffered = frame.to_buffered()
        for runtime in list(self.runtimes.values()):
            runtime.add_frame(buffered)

    # -- workflows ------------------------------------------------------

    def runtime_for(self, workflow: Workflow) -> WorkflowRuntime:
        existing = self.runtimes.get(workflow.id)
        if existing is not None:
            if existing.workflow.version != workflow.version:
                existing.apply_workflow_update(workflow)
            else:
                existing.workflow = workflow
            return existing

        runtime = WorkflowRuntime(
            workflow=workflow,
            db=self.db,
            dispatcher=self.dispatcher,
            evidence=self.evidence,
        )
        self.runtimes[workflow.id] = runtime
        return runtime

    def start_worker(self, workflow: Workflow) -> None:
        runtime = self.runtime_for(workflow)
        worker = self.workers.get(workflow.id)
        if worker is None:
            worker = InferenceWorker(
                runtime, self.capture, self.provider,
                interval_seconds=self.settings.sample_interval_seconds,
                concurrency=self._concurrency,
            )
            self.workers[workflow.id] = worker
        worker.start()

    def worker_running(self, workflow_id: str) -> bool:
        worker = self.workers.get(workflow_id)
        return bool(worker and worker._task and not worker._task.done())

    async def stop_worker(self, workflow_id: str) -> None:
        worker = self.workers.pop(workflow_id, None)
        if worker is not None:
            await worker.stop()

    async def shutdown(self) -> None:
        for workflow_id in list(self.workers):
            await self.stop_worker(workflow_id)
        self.capture.stop()
        self.db.close()


state: Optional[AppState] = None


def get_state() -> AppState:
    if state is None:  # pragma: no cover - guarded by lifespan
        raise HTTPException(503, "application is not initialised")
    return state


@asynccontextmanager
async def lifespan(app: FastAPI):
    global state
    settings = load_settings()
    state = AppState(settings)

    if not state.capture.start():
        log.error("capture failed to start for source %s",
                  state.capture.source.describe())

    # Resume anything that was enabled before the restart. Episode identity is
    # restored by the runtime, so an active condition does not re-fire.
    for workflow in state.db.list_workflows():
        if workflow.enabled:
            state.start_worker(workflow)
            log.info("resumed workflow %s v%s", workflow.id, workflow.version)

    try:
        yield
    finally:
        await state.shutdown()
        state = None


app = FastAPI(title="hacklanta-project camera agent", version="0.2.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------
# Health and cameras
# --------------------------------------------------------------------------


@app.get("/api/health")
def health() -> dict[str, Any]:
    st = get_state()
    return {
        "status": "ok",
        "camera": st.capture.status(),
        "vision_provider": st.settings.effective_vision_provider,
        "vision_model": (
            st.settings.vision_model
            if st.settings.effective_vision_provider == "gemini"
            else "heuristic-v1"
        ),
        "using_real_model": st.settings.effective_vision_provider == "gemini",
        "active_workflows": sum(
            1 for w in st.db.list_workflows() if w.enabled
        ),
        "sample_interval_seconds": st.settings.sample_interval_seconds,
        "webhook_destinations": st.settings.allowlist.ids(),
    }


@app.get("/api/cameras")
def list_cameras() -> list[dict[str, Any]]:
    st = get_state()
    status = st.capture.status()
    return [{
        "id": "cam_01",
        "source_kind": st.settings.camera_source_kind.value,
        # Anything that is not a live device is replay, and is labelled so.
        "is_live": st.settings.camera_source_kind == SourceKind.USB,
        **status,
    }]


@app.get("/api/cameras/{camera_id}/snapshot")
def camera_snapshot(camera_id: str) -> Response:
    st = get_state()
    jpeg = st.capture.latest_jpeg()
    if jpeg is None:
        raise HTTPException(503, "no frame available")
    return Response(content=jpeg, media_type="image/jpeg",
                    headers={"Cache-Control": "no-store"})


@app.get("/api/cameras/{camera_id}/preview")
def camera_preview(camera_id: str) -> StreamingResponse:
    """MJPEG stream for the Monitor screen."""
    st = get_state()
    interval = 1.0 / max(1.0, st.settings.preview_fps)

    async def frames():
        while True:
            jpeg = st.capture.latest_jpeg()
            if jpeg is not None:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n"
                       b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n"
                       + jpeg + b"\r\n")
            await asyncio.sleep(interval)

    return StreamingResponse(
        frames(), media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.post("/api/cameras/{camera_id}/crop-preview")
def crop_preview(camera_id: str, roi: ROI) -> Response:
    """Exactly the crop the model will inspect, so the user sees it first."""
    st = get_state()
    frame = st.capture.latest()
    if frame is None:
        raise HTTPException(503, "no frame available")
    crop = crop_region(frame.image, roi)
    return Response(content=encode_jpeg(crop, 85), media_type="image/jpeg",
                    headers={"Cache-Control": "no-store"})


# --------------------------------------------------------------------------
# Workflows
# --------------------------------------------------------------------------


@app.post("/api/workflows/compile")
def compile_workflow(req: CompileRequest) -> dict[str, Any]:
    """Instruction -> validated draft. Never activates anything."""
    result = compile_instruction(
        req.instruction, camera_id=req.camera_id, roi=req.roi
    )
    return {
        "ok": result.ok,
        "draft": result.draft.model_dump(mode="json") if result.draft else None,
        "questions": result.questions,
        "unsupported_reason": result.unsupported_reason,
        "preview_text": result.preview_text,
        "requires_preview": result.requires_preview,
    }


@app.get("/api/workflows")
def list_workflows() -> list[dict[str, Any]]:
    st = get_state()
    return [
        {**w.model_dump(mode="json"), "preview_text": build_preview_text(w)}
        for w in st.db.list_workflows()
    ]


@app.post("/api/workflows")
async def save_workflow(req: SaveWorkflowRequest) -> dict[str, Any]:
    """Save a workflow, starting its worker if it is enabled.

    Async on purpose: starting a worker calls ``asyncio.create_task``, which
    needs a running event loop. A sync def here would be handed to FastAPI's
    threadpool, where there is none.
    """
    st = get_state()
    workflow = req.workflow
    st.db.save_workflow(workflow)
    if workflow.enabled:
        st.start_worker(workflow)
    return {**workflow.model_dump(mode="json"),
            "preview_text": build_preview_text(workflow),
            "worker_running": st.worker_running(workflow.id)}


@app.patch("/api/workflows/{workflow_id}")
async def patch_workflow(workflow_id: str, req: PatchWorkflowRequest) -> dict[str, Any]:
    """Edit a workflow. Any change to the rule itself bumps the version.

    Version bumping is what invalidates in-flight results, so an edit can never
    be satisfied by evidence gathered under the previous rule.
    """
    st = get_state()
    current = st.db.get_workflow(workflow_id)
    if current is None:
        raise HTTPException(404, f"workflow {workflow_id} not found")

    changes = req.model_dump(exclude_none=True)
    if not changes:
        raise HTTPException(400, "no fields to update")

    rule_fields = {"target_description", "roi", "duration_seconds", "min_samples",
                   "max_gap_seconds", "clear_samples"}
    rule_changed = bool(rule_fields & set(changes))

    updated = current.model_copy(update=changes)
    if rule_changed:
        updated = updated.model_copy(update={"version": current.version + 1})

    # Re-validate: model_copy skips validators, so a bad edit would otherwise
    # be persisted unchecked.
    updated = Workflow.model_validate(updated.model_dump())

    st.db.save_workflow(updated)
    if updated.enabled:
        st.runtime_for(updated)
        st.start_worker(updated)
    else:
        await st.stop_worker(workflow_id)
        if workflow_id in st.runtimes:
            st.runtimes[workflow_id].workflow = updated

    return {**updated.model_dump(mode="json"),
            "preview_text": build_preview_text(updated),
            "version_bumped": rule_changed}


@app.get("/api/workflows/{workflow_id}/status")
def workflow_status(workflow_id: str) -> dict[str, Any]:
    st = get_state()
    workflow = st.db.get_workflow(workflow_id)
    if workflow is None:
        raise HTTPException(404, f"workflow {workflow_id} not found")

    runtime = st.runtimes.get(workflow_id)
    worker_running = st.worker_running(workflow_id)
    base: dict[str, Any] = {
        "workflow": workflow.model_dump(mode="json"),
        "preview_text": build_preview_text(workflow),
        "camera": st.capture.status(),
        "worker_running": worker_running,
        "samples_taken": st.workers[workflow_id].samples if workflow_id in st.workers else 0,
    }
    if runtime is None:
        status = {
            "episode_state": "IDLE", "condition_unverified": False,
            "positive_samples": 0, "clear_streak": 0,
            "camera_health": st.capture.health().value,
            "latest_observation": st.db.latest_observation(workflow_id),
        }
    else:
        status = runtime.status()

    base.update(status)
    base["display_state"] = _display_state(
        workflow, status, st.capture.health(), worker_running
    )
    return base


def _display_state(
    workflow: Workflow,
    status: dict,
    health: CameraHealth,
    worker_running: bool,
) -> str:
    """One label the UI shows.

    A fluid preview must never imply a fresh verdict, so this reports on the
    inference path rather than the camera. An enabled rule whose worker is not
    running, or which has not produced an observation yet, says so explicitly
    instead of claiming to be monitoring.
    """
    if not workflow.enabled:
        return "paused"
    if not worker_running:
        return "not running"
    if health == CameraHealth.DISCONNECTED:
        return "disconnected"

    latest = status.get("latest_observation")
    if not latest:
        return "starting"
    if latest.get("result") == "unknown":
        return "uncertain"
    if status.get("episode_state") == "ACTIVE":
        return "unverified" if status.get("condition_unverified") else "event active"
    if status.get("episode_state") == "CANDIDATE":
        return "confirming"
    if health == CameraHealth.STALE:
        return "stale"
    return "monitoring"


# --------------------------------------------------------------------------
# Events, tasks, actions
# --------------------------------------------------------------------------


@app.get("/api/events")
def list_events(
    workflow_id: Optional[str] = None,
    limit: int = Query(default=50, ge=1, le=500),
) -> list[dict[str, Any]]:
    st = get_state()
    events = st.db.list_events(workflow_id=workflow_id, limit=limit)
    for event in events:
        event["actions"] = [
            a.model_dump(mode="json") for a in st.db.list_actions(event["id"])
        ]
    return events


@app.get("/api/events/{event_id}")
def get_event(event_id: str) -> dict[str, Any]:
    st = get_state()
    event = st.db.get_event(event_id)
    if event is None:
        raise HTTPException(404, f"event {event_id} not found")
    event["actions"] = [
        a.model_dump(mode="json") for a in st.db.list_actions(event_id)
    ]
    return event


@app.get("/api/tasks")
def list_tasks(completed: Optional[bool] = None) -> list[dict[str, Any]]:
    return get_state().db.list_tasks(completed=completed)


@app.post("/api/actions/{action_id}/retry")
async def retry_action(action_id: str) -> dict[str, Any]:
    st = get_state()
    action = st.db.get_action(action_id)
    if action is None:
        raise HTTPException(404, f"action {action_id} not found")
    event = st.db.get_event(action.event_id)
    if event is None:
        raise HTTPException(404, "event for this action not found")
    workflow = st.db.get_workflow(event["workflow_id"])
    if workflow is None:
        raise HTTPException(404, "workflow for this action not found")

    updated = await st.dispatcher.retry(
        action_id, workflow.action, workflow_id=workflow.id
    )
    if updated is None:
        raise HTTPException(404, f"action {action_id} not found")
    return updated.model_dump(mode="json")


@app.get("/api/evidence/{filename}")
def get_evidence(filename: str) -> FileResponse:
    """Serve one evidence file.

    The name is resolved inside the evidence directory and the result must
    still be inside it, so '../' cannot escape.
    """
    st = get_state()
    base = st.settings.evidence_dir.resolve()
    target = (base / filename).resolve()
    if base != target.parent or not target.is_file():
        raise HTTPException(404, "evidence not found")
    media = "video/mp4" if target.suffix == ".mp4" else "image/jpeg"
    return FileResponse(target, media_type=media)


# --------------------------------------------------------------------------
# History
# --------------------------------------------------------------------------


@app.post("/api/history/query")
def history_query(req: HistoryQueryRequest) -> dict[str, Any]:
    st = get_state()
    result = answer_question(st.db, req.question, workflow_id=req.workflow_id)
    return {
        "supported": result.supported,
        "kind": result.kind,
        "answer": result.answer,
        "events": result.events,
        "actions": result.actions,
        "scope_note": result.scope_note,
        "suggestions": result.suggestions or (
            SUPPORTED_EXAMPLES if not result.supported else []
        ),
    }


# --------------------------------------------------------------------------
# Built frontend
# --------------------------------------------------------------------------

# Serving the compiled UI from the backend means a demo machine needs Python
# only -- no Node, no second process, no port proxying. `npm run dev` is still
# the right loop while developing the frontend; this is for running it.
#
# Mounted last on purpose: FastAPI matches in declaration order, so every
# /api/... route above wins before this catch-all sees the request.
FRONTEND_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"
if FRONTEND_DIST.is_dir():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="ui")
    log.info("serving built UI from %s", FRONTEND_DIST)
else:
    log.info("no built UI at %s; run `npm run build` in frontend/", FRONTEND_DIST)
