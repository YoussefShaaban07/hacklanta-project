"""Shared contracts for the camera agent.

Owner: C. Reviewed by A (frame metadata), B (AI response fields), D (UI needs).

These models are the frozen interface between the four workstreams. Changing a
field here is a coordinated change: update fixtures/ in the same commit so the
other owners' tests fail loudly rather than silently drifting.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


def utcnow() -> datetime:
    """UTC timestamp for stored records. Live durations use monotonic time."""
    return datetime.now(timezone.utc)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


# --------------------------------------------------------------------------
# Enumerations
# --------------------------------------------------------------------------


class ConditionType(str, Enum):
    """Condition templates.

    The first three are calibrated starter templates. ``visual_condition``
    stores other user-defined, visibly testable checks and must pass a
    positive/negative preview before activation (guide step 5).
    """

    REGION_EMPTY = "region_empty"
    OBJECT_PRESENT = "object_present"
    PERSON_PRESENT = "person_present"
    VISUAL_CONDITION = "visual_condition"


CALIBRATED_TEMPLATES = frozenset(
    {
        ConditionType.REGION_EMPTY,
        ConditionType.OBJECT_PRESENT,
        ConditionType.PERSON_PRESENT,
    }
)


class ObservationResult(str, Enum):
    MET = "met"
    NOT_MET = "not_met"
    #: Obscured, dark, blurred, ambiguous, or the model/transport failed.
    UNKNOWN = "unknown"


class EpisodeState(str, Enum):
    IDLE = "IDLE"
    CANDIDATE = "CANDIDATE"
    ACTIVE = "ACTIVE"


class CameraHealth(str, Enum):
    """Camera health is tracked separately from episode state (guide step 6)."""

    OK = "ok"
    STALE = "stale"
    DISCONNECTED = "disconnected"


class ActionType(str, Enum):
    CREATE_TASK = "create_task"
    WEBHOOK = "webhook"


class ActionStatus(str, Enum):
    PENDING = "pending"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class EventStatus(str, Enum):
    #: Episode still open and the condition is still being confirmed.
    ACTIVE = "active"
    #: Condition verified continuously but the camera has since gone unknown.
    UNVERIFIED = "unverified"
    #: Episode closed after clear_samples consecutive not_met results.
    CLOSED = "closed"


class SourceKind(str, Enum):
    """Pluggable frame sources. ``usb`` is the demo-laptop path; ``file`` is the
    repeatable test path; ``synthetic`` backs the tests with no I/O at all."""

    USB = "usb"
    FILE = "file"
    SYNTHETIC = "synthetic"
    RTSP = "rtsp"  # stretch input, not a supported promise


# --------------------------------------------------------------------------
# Region of interest
# --------------------------------------------------------------------------

UnitFloat = Annotated[float, Field(ge=0.0, le=1.0)]


class ROI(BaseModel):
    """Region of interest, normalized 0..1 against the full frame.

    Normalized coordinates survive preview letterboxing and resolution changes,
    which is why the UI converts pointer positions into image space rather than
    using the HTML element rectangle (guide step 9).
    """

    model_config = ConfigDict(extra="forbid")

    x: UnitFloat
    y: UnitFloat
    w: Annotated[float, Field(gt=0.0, le=1.0)]
    h: Annotated[float, Field(gt=0.0, le=1.0)]

    @model_validator(mode="after")
    def _within_bounds(self) -> "ROI":
        if self.x + self.w > 1.0 + 1e-9:
            raise ValueError(f"ROI extends past right edge: x+w={self.x + self.w:.4f}")
        if self.y + self.h > 1.0 + 1e-9:
            raise ValueError(f"ROI extends past bottom edge: y+h={self.y + self.h:.4f}")
        return self

    def to_pixels(self, width: int, height: int) -> tuple[int, int, int, int]:
        """Return an integer (x, y, w, h) crop box clamped inside the frame."""
        px = max(0, min(width - 1, round(self.x * width)))
        py = max(0, min(height - 1, round(self.y * height)))
        pw = max(1, min(width - px, round(self.w * width)))
        ph = max(1, min(height - py, round(self.h * height)))
        return px, py, pw, ph


# --------------------------------------------------------------------------
# Actions
# --------------------------------------------------------------------------


class CreateTaskAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal[ActionType.CREATE_TASK] = ActionType.CREATE_TASK
    title: Annotated[str, Field(min_length=1, max_length=200)]


class WebhookAction(BaseModel):
    """The compiler picks a *destination id* from a backend allowlist.

    The language model never supplies a URL (guide step 7): it chooses an id,
    and the backend resolves it. This keeps prompt-injected text from steering
    outbound requests.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal[ActionType.WEBHOOK] = ActionType.WEBHOOK
    destination_id: Annotated[str, Field(min_length=1, max_length=64)]
    title: Annotated[str, Field(min_length=1, max_length=200)] = "Camera event"


Action = Annotated[CreateTaskAction | WebhookAction, Field(discriminator="type")]


# --------------------------------------------------------------------------
# Workflow
# --------------------------------------------------------------------------


class Workflow(BaseModel):
    """A camera + visible condition + temporal confirmation + software action.

    Editing a workflow increases ``version`` and resets its timer; results
    carrying an older version are rejected by the engine.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: new_id("wf"))
    version: Annotated[int, Field(ge=1)] = 1
    camera_id: str = "cam_01"

    condition_type: ConditionType
    target_description: Annotated[str, Field(min_length=1, max_length=500)]
    roi: ROI

    #: Wall-clock seconds the condition must be sustained across samples.
    duration_seconds: Annotated[float, Field(gt=0, le=3600)] = 10.0
    #: Positive samples required in addition to the duration.
    min_samples: Annotated[int, Field(ge=1, le=100)] = 3
    #: A gap longer than this breaks the streak and resets the candidate.
    max_gap_seconds: Annotated[float, Field(gt=0, le=600)] = 5.0
    #: Consecutive fresh not_met results needed to close an active episode.
    clear_samples: Annotated[int, Field(ge=1, le=100)] = 3

    action: Action
    enabled: bool = False

    #: Reference image for "this tray" style descriptions. Not identity tracking.
    reference_image_path: Optional[str] = None
    #: A user-defined visual_condition must pass a preview before activation.
    preview_validated: bool = False

    @model_validator(mode="after")
    def _guard_activation(self) -> "Workflow":
        if (
            self.enabled
            and self.condition_type == ConditionType.VISUAL_CONDITION
            and not self.preview_validated
        ):
            raise ValueError(
                "visual_condition must pass a positive/negative preview test "
                "before it can be enabled"
            )
        return self


# --------------------------------------------------------------------------
# Observation
# --------------------------------------------------------------------------


class Observation(BaseModel):
    """One model verdict about one frame, for one workflow version."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: new_id("obs"))
    workflow_id: str
    workflow_version: Annotated[int, Field(ge=1)]
    camera_id: str
    frame_id: Annotated[int, Field(ge=0)]

    captured_at_utc: datetime
    processed_at_utc: datetime = Field(default_factory=utcnow)

    #: Monotonic seconds at capture. The engine measures live durations with
    #: this; UTC is for stored records and display only (guide step 6).
    captured_monotonic: float

    result: ObservationResult
    explanation: str = ""
    model_id: str = ""
    inference_ms: Annotated[int, Field(ge=0)] = 0
    evidence_path: Optional[str] = None

    @property
    def is_positive(self) -> bool:
        return self.result == ObservationResult.MET


# --------------------------------------------------------------------------
# Episode / Event / Action records
# --------------------------------------------------------------------------


class Episode(BaseModel):
    """Persistent runtime identity that deduplicates events.

    One episode produces at most one event. ``(workflow_id, episode_id)`` is
    unique in the events table, so a crash mid-commit cannot double-fire.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: new_id("ep"))
    workflow_id: str
    workflow_version: Annotated[int, Field(ge=1)]
    state: EpisodeState = EpisodeState.IDLE

    #: Monotonic seconds of the first positive sample in the current streak.
    first_positive_monotonic: Optional[float] = None
    first_positive_at_utc: Optional[datetime] = None
    #: Monotonic seconds of the most recent positive sample.
    last_positive_monotonic: Optional[float] = None

    positive_samples: int = 0
    clear_streak: int = 0

    #: True once an ACTIVE episode sees unknown or loses the camera. The episode
    #: keeps its identity but must not re-arm on that evidence.
    condition_unverified: bool = False

    #: Highest frame_id already folded in; guards against out-of-order arrival.
    last_frame_id: int = -1


class Event(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: new_id("evt"))
    workflow_id: str
    workflow_version: Annotated[int, Field(ge=1)] = 1
    episode_id: str
    first_positive_at: datetime
    triggered_at: datetime
    status: EventStatus = EventStatus.ACTIVE
    snapshot_path: Optional[str] = None
    clip_path: Optional[str] = None
    explanation: str = ""


class ActionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: new_id("act"))
    event_id: str
    type: ActionType
    status: ActionStatus = ActionStatus.PENDING
    attempts: int = 0
    last_error: Optional[str] = None
    receipt: Optional[str] = None
    #: Stable key so a replayed delivery is deduplicated receiver-side.
    idempotency_key: str = ""
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


class Task(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: new_id("task"))
    title: str
    workflow_id: str
    event_id: str
    created_at: datetime = Field(default_factory=utcnow)
    completed: bool = False
    evidence_path: Optional[str] = None


# --------------------------------------------------------------------------
# Engine effects
# --------------------------------------------------------------------------


class EffectKind(str, Enum):
    NONE = "none"
    TRIGGER_EVENT = "trigger_event"
    CLOSE_EPISODE = "close_episode"
    #: Observation rejected: stale frame, duplicate, or superseded version.
    REJECTED = "rejected"


class Effect(BaseModel):
    """What the engine decided. Pure data so transitions are unit-testable."""

    model_config = ConfigDict(extra="forbid")

    kind: EffectKind = EffectKind.NONE
    episode_id: Optional[str] = None
    reason: str = ""
