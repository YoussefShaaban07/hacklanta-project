"""Workflow runtime: the seam between observations, the engine and effects.

Owner: C, integrating A's evidence and B's observations.

This is deliberately thin. All the judgement lives in :mod:`engine` (pure) and
all the durability lives in :mod:`db` (transactional). This module only
sequences them, which is why it can be exercised end to end in a test with no
camera, no model and no network.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Optional

from .actions import Dispatcher, build_action_record
from .db import Database
from .engine import on_camera_unhealthy, restore_after_restart, transition
from .evidence import BufferedFrame, EvidenceStore, RollingBuffer
from .schemas import (
    ActionStatus,
    ActionType,
    CameraHealth,
    Effect,
    EffectKind,
    Episode,
    EpisodeState,
    Event,
    EventStatus,
    Observation,
    Workflow,
    utcnow,
)

log = logging.getLogger(__name__)

#: Clip span around the trigger, in seconds.
CLIP_BEFORE = 5.0
CLIP_AFTER = 5.0


@dataclass
class IngestResult:
    effect: Effect
    event_id: Optional[str] = None
    created_task: bool = False
    rejected: bool = False


@dataclass
class WorkflowRuntime:
    """Owns one workflow's live episode and drives its effects."""

    workflow: Workflow
    db: Database
    dispatcher: Dispatcher
    evidence: EvidenceStore
    buffer: RollingBuffer = field(default_factory=RollingBuffer)
    episode: Episode = field(init=False)
    camera_health: CameraHealth = CameraHealth.OK
    #: Events whose clip window has not closed yet.
    _pending_clips: dict[str, float] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        stored = self.db.load_episodes(self.workflow.id)
        if stored:
            self.episode = restore_after_restart(stored[0])
        else:
            self.episode = Episode(
                workflow_id=self.workflow.id,
                workflow_version=self.workflow.version,
            )
        self.db.save_episode(self.episode)

    # -- workflow versioning ---------------------------------------------

    def apply_workflow_update(self, updated: Workflow) -> None:
        """Adopt a new rule version and reset the timer.

        In-flight observations carrying the old version are rejected by the
        engine, so there is no window where a half-edited rule can fire.
        """
        if updated.id != self.workflow.id:
            raise ValueError("cannot apply a different workflow id")
        self.workflow = updated
        self.episode = Episode(
            workflow_id=updated.id, workflow_version=updated.version
        )
        self.db.save_workflow(updated)
        self.db.save_episode(self.episode)

    # -- ingest -----------------------------------------------------------

    async def ingest(
        self, obs: Observation, frame_jpeg: Optional[bytes] = None
    ) -> IngestResult:
        """Fold one observation in, persist it, and carry out the effect."""
        self.db.save_observation(obs)

        decision = transition(self.episode, obs, self.workflow)
        previous_episode_id = self.episode.id
        self.episode = decision.episode
        effect = decision.effect

        if effect.kind == EffectKind.REJECTED:
            log.debug("rejected observation %s: %s", obs.id, effect.reason)
            return IngestResult(effect=effect, rejected=True)

        self.db.save_episode(self.episode)

        if effect.kind == EffectKind.TRIGGER_EVENT:
            return await self._on_trigger(obs, effect, frame_jpeg)

        if effect.kind == EffectKind.CLOSE_EPISODE:
            self.db.close_episode_event(self.workflow.id, effect.episode_id)
            log.info("episode %s closed: %s", effect.episode_id, effect.reason)

        return IngestResult(effect=effect)

    async def _on_trigger(
        self, obs: Observation, effect: Effect, frame_jpeg: Optional[bytes]
    ) -> IngestResult:
        event = Event(
            workflow_id=self.workflow.id,
            workflow_version=self.workflow.version,
            episode_id=effect.episode_id,
            first_positive_at=self.episode.first_positive_at_utc or obs.captured_at_utc,
            triggered_at=obs.captured_at_utc,
            status=EventStatus.ACTIVE,
            explanation=obs.explanation or effect.reason,
        )

        # Snapshot first so evidence exists before anything is dispatched.
        snapshot = frame_jpeg
        if snapshot is None:
            latest = self.buffer.latest()
            snapshot = latest.jpeg if latest else None
        if snapshot is not None:
            try:
                event.snapshot_path = self.evidence.write_snapshot(event.id, snapshot)
            except Exception:  # evidence must never block the action
                log.exception("snapshot failed for event %s", event.id)

        action_type = ActionType(self.workflow.action.type)
        action = build_action_record(event.id, action_type)

        # Only the event and its *pending* action commit together. Carrying out
        # the action -- creating the task, delivering the webhook -- is the
        # dispatcher's job, so there is exactly one place that performs an
        # effect and exactly one place that records its receipt. If this
        # process dies before dispatch, the pending row is the recovery point.
        committed = self.db.commit_event_with_action(event, action)
        if not committed:
            # The episode already produced an event. Normal after a crash-retry.
            log.info("episode %s already has an event; not firing again", effect.episode_id)
            return IngestResult(effect=effect, event_id=None)

        self._pending_clips[event.id] = obs.captured_monotonic

        done = await self.dispatcher.execute(
            action, self.workflow.action, workflow_id=self.workflow.id
        )
        return IngestResult(
            effect=effect,
            event_id=event.id,
            created_task=(
                action_type == ActionType.CREATE_TASK
                and done.status == ActionStatus.SUCCEEDED
            ),
        )

    # -- evidence ---------------------------------------------------------

    def add_frame(self, frame: BufferedFrame) -> None:
        self.buffer.add(frame)

    def finalize_ready_clips(self) -> list[str]:
        """Encode clips whose trailing window has filled.

        Called from a worker, never from capture: encoding is slow and must not
        stall frame reads.
        """
        finished: list[str] = []
        for event_id, center in list(self._pending_clips.items()):
            if not self.buffer.has_covered(center, CLIP_AFTER):
                continue
            frames = self.buffer.window(center, CLIP_BEFORE, CLIP_AFTER)
            path = self.evidence.write_clip(event_id, frames)
            if path:
                self.db.attach_evidence(event_id, clip_path=path)
                finished.append(event_id)
            self._pending_clips.pop(event_id, None)
        return finished

    # -- health -----------------------------------------------------------

    def set_camera_health(self, health: CameraHealth) -> None:
        """Camera health is tracked apart from episode state."""
        if health == self.camera_health:
            return
        self.camera_health = health
        if health == CameraHealth.OK:
            return

        decision = on_camera_unhealthy(self.episode)
        self.episode = decision.episode
        self.db.save_episode(self.episode)

        # An active episode held through a camera loss is recorded as
        # unverified so the UI does not present it as currently confirmed.
        if self.episode.state == EpisodeState.ACTIVE and self.episode.condition_unverified:
            rows = self.db.list_events(workflow_id=self.workflow.id, limit=1)
            if rows and rows[0]["episode_id"] == self.episode.id:
                self.db.set_event_status(rows[0]["id"], EventStatus.UNVERIFIED)

    # -- status -----------------------------------------------------------

    def status(self) -> dict:
        """What the Monitor screen polls once a second."""
        latest = self.db.latest_observation(self.workflow.id)
        return {
            "workflow_id": self.workflow.id,
            "workflow_version": self.workflow.version,
            "enabled": self.workflow.enabled,
            "episode_state": self.episode.state.value,
            "condition_unverified": self.episode.condition_unverified,
            "positive_samples": self.episode.positive_samples,
            "clear_streak": self.episode.clear_streak,
            "camera_health": self.camera_health.value,
            "latest_observation": latest,
            "buffered_frames": len(self.buffer),
            "pending_clips": len(self._pending_clips),
        }
