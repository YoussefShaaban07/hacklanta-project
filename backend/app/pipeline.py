"""Inference pipeline: frames -> vision -> engine.

Owner: B and C jointly. Sampling policy, in one place:

  * Preview runs fast; analysis runs slow. They are decoupled so a fluid feed
    can never imply a fresh verdict.
  * Exactly one inference per workflow is in flight at a time. If a call takes
    longer than the sample interval, the next tick is skipped rather than
    queued -- old pending frames are dropped, never backlogged.
  * Blocking provider calls run in a thread so the event loop keeps serving
    HTTP while a model request is outstanding.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

from .camera import CaptureWorker, crop_region
from .evidence import encode_jpeg
from .runtime import WorkflowRuntime
from .schemas import CameraHealth, Observation, ObservationResult, utcnow
from .vision import VisionProvider

log = logging.getLogger(__name__)


class InferenceWorker:
    """Samples the latest frame for one workflow and feeds the engine."""

    def __init__(
        self,
        runtime: WorkflowRuntime,
        capture: CaptureWorker,
        provider: VisionProvider,
        *,
        interval_seconds: float = 2.0,
        concurrency: Optional[asyncio.Semaphore] = None,
    ) -> None:
        self.runtime = runtime
        self.capture = capture
        self.provider = provider
        self.interval = max(0.2, interval_seconds)
        self._concurrency = concurrency or asyncio.Semaphore(2)
        self._task: Optional[asyncio.Task] = None
        self._stop = asyncio.Event()
        self._last_frame_id = -1
        self.samples = 0

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._stop.clear()
            self._task = asyncio.create_task(self._run(), name=f"infer:{self.runtime.workflow.id}")

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=5.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._task.cancel()
            self._task = None

    async def _run(self) -> None:
        while not self._stop.is_set():
            started = time.monotonic()
            try:
                await self.tick()
            except Exception:
                log.exception("inference tick failed")
            elapsed = time.monotonic() - started
            try:
                await asyncio.wait_for(
                    self._stop.wait(), timeout=max(0.0, self.interval - elapsed)
                )
            except asyncio.TimeoutError:
                pass

    async def tick(self) -> Optional[Observation]:
        """One sample. Returns the observation, or None if nothing to do."""
        workflow = self.runtime.workflow

        health = self.capture.health()
        self.runtime.set_camera_health(health)
        if health == CameraHealth.DISCONNECTED:
            return None

        if not workflow.enabled:
            return None

        frame = self.capture.latest()
        if frame is None:
            return None

        # Never analyse the same frame twice: a duplicate verdict would advance
        # the sample count without new evidence.
        if frame.frame_id == self._last_frame_id:
            return None
        self._last_frame_id = frame.frame_id

        crop = crop_region(frame.image, workflow.roi)
        try:
            crop_jpeg = encode_jpeg(crop, 85)
        except Exception:
            log.exception("could not encode crop")
            return None

        loop = asyncio.get_running_loop()
        async with self._concurrency:
            outcome = await loop.run_in_executor(
                None, self.provider.analyze, crop_jpeg, workflow
            )

        self.samples += 1
        observation = Observation(
            workflow_id=workflow.id,
            workflow_version=workflow.version,
            camera_id=workflow.camera_id,
            frame_id=frame.frame_id,
            captured_at_utc=frame.captured_at_utc,
            processed_at_utc=utcnow(),
            captured_monotonic=frame.captured_monotonic,
            result=outcome.result,
            explanation=outcome.explanation,
            model_id=outcome.model_id,
            inference_ms=outcome.inference_ms,
        )

        try:
            frame_jpeg = encode_jpeg(frame.image, 85)
        except Exception:
            frame_jpeg = None

        await self.runtime.ingest(observation, frame_jpeg=frame_jpeg)
        self.runtime.finalize_ready_clips()
        return observation
