"""End-to-end spine test.

Synthetic frames in, real engine, real SQLite, real JPEG and MP4 encoding, with
only the outbound HTTP boundary mocked. This is the guide's definition of done
reduced to something that runs in CI: cause a condition, get exactly one
action, open its evidence, and still have it after a restart.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import cv2
import httpx
import numpy as np
import pytest

from app.actions import DestinationAllowlist, Dispatcher, WebhookDestination
from app.db import Database
from app.evidence import BufferedFrame, EvidenceStore, RollingBuffer, encode_jpeg
from app.runtime import WorkflowRuntime
from app.schemas import (
    ActionStatus,
    CameraHealth,
    EpisodeState,
    EventStatus,
    ObservationResult,
    WebhookAction,
)

from conftest import BASE_MONO, BASE_UTC, Clock, make_workflow

MET = ObservationResult.MET
NOT_MET = ObservationResult.NOT_MET
UNKNOWN = ObservationResult.UNKNOWN


def make_frame(label: int, width: int = 320, height: int = 240) -> np.ndarray:
    """A distinguishable frame, so a clip is visibly not all one colour."""
    frame = np.full((height, width, 3), (label * 7) % 255, dtype=np.uint8)
    cv2.putText(
        frame, f"f{label}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2
    )
    return frame


@pytest.fixture
def stack(tmp_path):
    """A fully wired runtime with an in-test webhook receiver."""
    db = Database(tmp_path / "e2e.db")
    workflow = make_workflow(
        action=WebhookAction(destination_id="test_receiver", title="Restock"),
    )
    db.save_workflow(workflow)

    received: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        body = json.loads(request.read().decode())
        # A correct receiver deduplicates on the idempotency key.
        if any(r["idempotency_key"] == body["idempotency_key"] for r in received):
            return httpx.Response(200, text='{"deduplicated":true}')
        received.append(body)
        return httpx.Response(201, text='{"ok":true}')

    allowlist = DestinationAllowlist()
    allowlist.add(WebhookDestination(id="test_receiver", url="https://receiver.test/h"))
    dispatcher = Dispatcher(
        db, allowlist, transport=httpx.MockTransport(handler)
    )
    runtime = WorkflowRuntime(
        workflow=workflow,
        db=db,
        dispatcher=dispatcher,
        evidence=EvidenceStore(tmp_path / "evidence"),
        buffer=RollingBuffer(seconds=30.0, fps=5.0),
    )
    yield runtime, db, received, tmp_path
    db.close()


def feed_frame(runtime: WorkflowRuntime, idx: int, at: float) -> bytes:
    jpeg = encode_jpeg(make_frame(idx))
    runtime.add_frame(
        BufferedFrame(
            frame_id=idx,
            jpeg=jpeg,
            captured_monotonic=BASE_MONO + at,
            captured_at_utc=BASE_UTC + timedelta(seconds=at),
        )
    )
    return jpeg


class TestHappyPath:
    async def test_condition_produces_one_event_with_evidence(self, stack):
        runtime, db, received, tmp_path = stack
        clock = Clock(runtime.workflow)

        results = []
        for i, t in enumerate(range(0, 22, 2)):
            jpeg = feed_frame(runtime, i, float(t))
            res = await runtime.ingest(
                clock.obs(MET, float(t), explanation="tray is empty"), frame_jpeg=jpeg
            )
            results.append(res)

        fired = [r for r in results if r.event_id]
        assert len(fired) == 1, "exactly one event for one sustained condition"

        event = db.get_event(fired[0].event_id)
        assert event["status"] == EventStatus.ACTIVE.value
        assert event["explanation"] == "tray is empty"

        # Evidence exists on disk, not just as a path in a row.
        assert Path(event["snapshot_path"]).exists()
        assert Path(event["snapshot_path"]).stat().st_size > 0

        # The action was delivered and the receipt recorded.
        actions = db.list_actions(event["id"])
        assert len(actions) == 1
        assert actions[0].status == ActionStatus.SUCCEEDED
        assert len(received) == 1
        assert received[0]["event_id"] == event["id"]

    async def test_clip_encodes_from_real_timestamps(self, stack):
        runtime, db, _received, _tmp = stack
        clock = Clock(runtime.workflow)

        event_id = None
        for i, t in enumerate(range(0, 22, 2)):
            jpeg = feed_frame(runtime, i, float(t))
            res = await runtime.ingest(clock.obs(MET, float(t)), frame_jpeg=jpeg)
            if res.event_id:
                event_id = res.event_id

        assert event_id is not None
        # Trailing window has not closed on the trigger frame yet at first.
        for i, t in enumerate(range(22, 34, 2), start=11):
            feed_frame(runtime, i, float(t))

        finished = runtime.finalize_ready_clips()
        assert event_id in finished

        clip = db.get_event(event_id)["clip_path"]
        assert clip and Path(clip).exists() and Path(clip).stat().st_size > 0

        cap = cv2.VideoCapture(clip)
        try:
            assert cap.isOpened(), "written clip must be readable"
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            assert frames > 0
        finally:
            cap.release()


class TestDemoCriticalProperties:
    async def test_sustained_condition_never_duplicates_the_action(self, stack):
        """Five minutes of a true condition: one task, one webhook."""
        runtime, db, received, _tmp = stack
        clock = Clock(runtime.workflow)
        for i, t in enumerate(range(0, 300, 2)):
            jpeg = feed_frame(runtime, i, float(t))
            await runtime.ingest(clock.obs(MET, float(t)), frame_jpeg=jpeg)

        assert len(db.list_events(workflow_id=runtime.workflow.id)) == 1
        assert len(received) == 1

    async def test_brief_condition_produces_nothing(self, stack):
        runtime, db, received, _tmp = stack
        clock = Clock(runtime.workflow)
        for i, (res, t) in enumerate([(MET, 0), (MET, 2), (NOT_MET, 4)]):
            jpeg = feed_frame(runtime, i, float(t))
            await runtime.ingest(clock.obs(res, float(t)), frame_jpeg=jpeg)

        assert db.list_events(workflow_id=runtime.workflow.id) == []
        assert received == []
        assert db.list_tasks() == []

    async def test_clear_then_recur_fires_a_second_distinct_event(self, stack):
        runtime, db, received, _tmp = stack
        clock = Clock(runtime.workflow)
        script = (
            [(MET, t) for t in (0, 5, 10)]
            + [(NOT_MET, t) for t in (12, 14, 16)]
            + [(MET, t) for t in (20, 25, 30)]
        )
        for i, (res, t) in enumerate(script):
            jpeg = feed_frame(runtime, i, float(t))
            await runtime.ingest(clock.obs(res, float(t)), frame_jpeg=jpeg)

        events = db.list_events(workflow_id=runtime.workflow.id)
        assert len(events) == 2
        assert len({e["episode_id"] for e in events}) == 2
        assert len(received) == 2
        # The first event was closed when the condition cleared.
        statuses = sorted(e["status"] for e in events)
        assert EventStatus.CLOSED.value in statuses

    async def test_occlusion_never_confirms(self, stack):
        runtime, db, received, _tmp = stack
        clock = Clock(runtime.workflow)
        i = 0
        for cycle in range(30):
            for res, offset in ((MET, 0.0), (UNKNOWN, 1.0)):
                t = cycle * 2 + offset
                jpeg = feed_frame(runtime, i, t)
                await runtime.ingest(clock.obs(res, t), frame_jpeg=jpeg)
                i += 1

        assert db.list_events(workflow_id=runtime.workflow.id) == []
        assert received == []

    async def test_camera_loss_marks_active_event_unverified(self, stack):
        runtime, db, _received, _tmp = stack
        clock = Clock(runtime.workflow)
        event_id = None
        for i, t in enumerate((0, 5, 10)):
            jpeg = feed_frame(runtime, i, float(t))
            res = await runtime.ingest(clock.obs(MET, float(t)), frame_jpeg=jpeg)
            if res.event_id:
                event_id = res.event_id

        assert runtime.episode.state == EpisodeState.ACTIVE
        runtime.set_camera_health(CameraHealth.DISCONNECTED)

        assert runtime.episode.condition_unverified is True
        assert db.get_event(event_id)["status"] == EventStatus.UNVERIFIED.value
        assert runtime.status()["camera_health"] == "disconnected"

    async def test_workflow_edit_resets_and_old_results_cannot_fire(self, stack):
        runtime, db, received, _tmp = stack
        clock = Clock(runtime.workflow)

        for i, t in enumerate((0, 4)):
            jpeg = feed_frame(runtime, i, float(t))
            await runtime.ingest(clock.obs(MET, float(t)), frame_jpeg=jpeg)
        assert runtime.episode.state == EpisodeState.CANDIDATE

        updated = runtime.workflow.model_copy(update={"version": 2})
        runtime.apply_workflow_update(updated)
        assert runtime.episode.state == EpisodeState.IDLE, "edit resets the timer"

        # Results still in flight from v1 arrive and must all be rejected.
        for i, t in enumerate((8, 12, 16), start=2):
            jpeg = feed_frame(runtime, i, float(t))
            res = await runtime.ingest(
                clock.obs(MET, float(t), version=1), frame_jpeg=jpeg
            )
            assert res.rejected

        assert db.list_events(workflow_id=runtime.workflow.id) == []
        assert received == []


class TestRestartRecovery:
    async def test_history_survives_restart_without_refiring(self, stack):
        runtime, db, received, tmp_path = stack
        clock = Clock(runtime.workflow)
        workflow = runtime.workflow

        for i, t in enumerate((0, 5, 10)):
            jpeg = feed_frame(runtime, i, float(t))
            await runtime.ingest(clock.obs(MET, float(t)), frame_jpeg=jpeg)

        events_before = db.list_events(workflow_id=workflow.id)
        assert len(events_before) == 1
        assert len(received) == 1
        db.close()

        # --- restart: new Database, new runtime, same files ---
        db2 = Database(tmp_path / "e2e.db")
        allowlist = DestinationAllowlist()
        allowlist.add(
            WebhookDestination(id="test_receiver", url="https://receiver.test/h")
        )

        def handler(request: httpx.Request) -> httpx.Response:
            import json

            received.append(json.loads(request.read().decode()))
            return httpx.Response(201, text='{"ok":true}')

        runtime2 = WorkflowRuntime(
            workflow=workflow,
            db=db2,
            dispatcher=Dispatcher(
                db2, allowlist, transport=httpx.MockTransport(handler)
            ),
            evidence=EvidenceStore(tmp_path / "evidence"),
            buffer=RollingBuffer(seconds=30.0, fps=5.0),
        )

        assert runtime2.episode.state == EpisodeState.ACTIVE
        assert runtime2.episode.id == events_before[0]["episode_id"]

        # Condition still true after reboot: no second task, no second webhook.
        fresh = Clock(workflow)
        for i, t in enumerate((0, 5, 10, 15, 20)):
            jpeg = feed_frame(runtime2, i, float(t))
            await runtime2.ingest(fresh.obs(MET, float(t)), frame_jpeg=jpeg)

        assert len(db2.list_events(workflow_id=workflow.id)) == 1
        assert len(received) == 1, "restart must not re-deliver the action"

        # Evidence from before the restart is still openable.
        assert Path(events_before[0]["snapshot_path"]).exists()
        db2.close()


class TestStatus:
    async def test_status_reports_live_state(self, stack):
        runtime, _db, _received, _tmp = stack
        clock = Clock(runtime.workflow)
        for i, t in enumerate((0, 2)):
            jpeg = feed_frame(runtime, i, float(t))
            await runtime.ingest(clock.obs(MET, float(t)), frame_jpeg=jpeg)

        status = runtime.status()
        assert status["episode_state"] == EpisodeState.CANDIDATE.value
        assert status["positive_samples"] == 2
        assert status["camera_health"] == CameraHealth.OK.value
        assert status["latest_observation"]["result"] == MET.value
        assert status["buffered_frames"] == 2
