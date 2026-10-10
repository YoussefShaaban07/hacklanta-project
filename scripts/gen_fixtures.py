"""Emit the shared JSON fixtures from the live schemas.

Generated, never hand-written: a fixture that cannot round-trip through the
models is a contract drift the other owners would otherwise find at merge time.
"""
import json, sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, "backend")
from app.schemas import (  # noqa: E402
    ActionRecord, ActionType, ConditionType, CreateTaskAction, Episode,
    EpisodeState, Event, EventStatus, Observation, ObservationResult, ROI,
    Task, Workflow,
)

T0 = datetime(2026, 10, 9, 14, 30, 0, tzinfo=timezone.utc)
MONO = 48210.0

workflow = Workflow(
    id="wf_001", version=1, camera_id="cam_01",
    condition_type=ConditionType.REGION_EMPTY,
    target_description="supply tray contains no bottles",
    roi=ROI(x=0.1, y=0.2, w=0.5, h=0.4),
    duration_seconds=10.0, min_samples=3, max_gap_seconds=5.0, clear_samples=3,
    action=CreateTaskAction(title="Restock the supply tray"),
    enabled=True,
)
observation = Observation(
    id="obs_7c1d9a2b4e55", workflow_id="wf_001", workflow_version=1,
    camera_id="cam_01", frame_id=142,
    captured_at_utc=T0 + timedelta(seconds=10),
    processed_at_utc=T0 + timedelta(seconds=10, milliseconds=840),
    captured_monotonic=MONO + 10.0,
    result=ObservationResult.MET,
    explanation="The tray in the region is bare; no bottles are visible.",
    model_id="gemini-vision-placeholder", inference_ms=812,
    evidence_path="data/evidence/obs_7c1d9a2b4e55.jpg",
)
event = Event(
    id="evt_3f9b1c7d2a80", workflow_id="wf_001", workflow_version=1,
    episode_id="ep_55aa90c31b7e",
    first_positive_at=T0, triggered_at=T0 + timedelta(seconds=10),
    status=EventStatus.ACTIVE,
    snapshot_path="data/evidence/evt_3f9b1c7d2a80.jpg",
    clip_path="data/evidence/evt_3f9b1c7d2a80.mp4",
    explanation="Tray empty across 3 samples spanning 10.0s.",
)
episode = Episode(
    id="ep_55aa90c31b7e", workflow_id="wf_001", workflow_version=1,
    state=EpisodeState.ACTIVE,
    first_positive_monotonic=MONO, first_positive_at_utc=T0,
    last_positive_monotonic=MONO + 10.0,
    positive_samples=3, clear_streak=0, condition_unverified=False,
    last_frame_id=142,
)
action = ActionRecord(
    id="act_9d2e4b6a1c03", event_id="evt_3f9b1c7d2a80",
    type=ActionType.CREATE_TASK, attempts=1,
    receipt="task:task_11bb22cc33dd",
    idempotency_key="8f14e45fceea167a5a36dedd4bea2543",
    created_at=T0 + timedelta(seconds=10),
    updated_at=T0 + timedelta(seconds=10, milliseconds=120),
)
task = Task(
    id="task_11bb22cc33dd", title="Restock the supply tray",
    workflow_id="wf_001", event_id="evt_3f9b1c7d2a80",
    created_at=T0 + timedelta(seconds=10), completed=False,
    evidence_path="data/evidence/evt_3f9b1c7d2a80.jpg",
)

for name, model in [
    ("workflow", workflow), ("observation", observation), ("event", event),
    ("episode", episode), ("action", action), ("task", task),
]:
    path = f"fixtures/{name}.json"
    with open(path, "w") as fh:
        json.dump(model.model_dump(mode="json"), fh, indent=2)
        fh.write("\n")
    print("wrote", path)
