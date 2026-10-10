"""Shared test helpers.

The engine takes its timebase from the observation, so tests drive it with
synthetic timestamps and never sleep. A whole ten-second confirmation runs in
microseconds.
"""

from __future__ import annotations

import itertools
from datetime import datetime, timedelta, timezone

import pytest

from app.schemas import (
    ConditionType,
    CreateTaskAction,
    Episode,
    Observation,
    ObservationResult,
    ROI,
    Workflow,
)

BASE_UTC = datetime(2026, 10, 9, 12, 0, 0, tzinfo=timezone.utc)
#: Arbitrary monotonic origin. Real monotonic clocks start wherever they like,
#: so the tests deliberately do not start at zero.
BASE_MONO = 10_000.0


class Clock:
    """Mints observations at explicit offsets with auto-incrementing frame ids."""

    def __init__(self, workflow: Workflow) -> None:
        self.workflow = workflow
        self._frames = itertools.count()

    def obs(
        self,
        result: ObservationResult | str,
        at: float,
        *,
        frame_id: int | None = None,
        version: int | None = None,
        explanation: str = "",
    ) -> Observation:
        """``at`` is seconds since the start of the scenario."""
        if isinstance(result, str):
            result = ObservationResult(result)
        fid = next(self._frames) if frame_id is None else frame_id
        return Observation(
            workflow_id=self.workflow.id,
            workflow_version=self.workflow.version if version is None else version,
            camera_id=self.workflow.camera_id,
            frame_id=fid,
            captured_at_utc=BASE_UTC + timedelta(seconds=at),
            captured_monotonic=BASE_MONO + at,
            result=result,
            explanation=explanation,
            model_id="test-model",
            inference_ms=12,
        )


def make_workflow(**overrides) -> Workflow:
    """The guide's worked example: empty supply tray for 10s -> restock task."""
    defaults = dict(
        id="wf_001",
        version=1,
        camera_id="cam_01",
        condition_type=ConditionType.REGION_EMPTY,
        target_description="supply tray contains no bottles",
        roi=ROI(x=0.1, y=0.2, w=0.5, h=0.4),
        duration_seconds=10.0,
        min_samples=3,
        max_gap_seconds=5.0,
        clear_samples=3,
        action=CreateTaskAction(title="Restock the supply tray"),
        enabled=True,
    )
    defaults.update(overrides)
    return Workflow(**defaults)


@pytest.fixture
def workflow() -> Workflow:
    return make_workflow()


@pytest.fixture
def episode(workflow: Workflow) -> Episode:
    return Episode(workflow_id=workflow.id, workflow_version=workflow.version)


@pytest.fixture
def clock(workflow: Workflow) -> Clock:
    return Clock(workflow)
