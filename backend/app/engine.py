"""Event state machine.

Owner: C. This module decides when repeated model verdicts amount to a real
event. It is deliberately pure: ``transition`` takes an episode plus one
observation and returns a new episode and an :class:`Effect`. No I/O, no
clocks, no database. The caller persists the result and performs the effect.

That purity is the point. Every row of the guide's state table, and every row
of its acceptance scenario table, is reachable from synthetic timestamped
observations in a unit test without a camera or a model.

Two timebases, deliberately:
  * ``captured_monotonic`` drives all duration arithmetic. A wall clock that
    steps backwards (NTP, DST) must never retroactively satisfy a timer.
  * ``captured_at_utc`` is carried into stored records for display and query.
"""

from __future__ import annotations

from dataclasses import dataclass

from .schemas import (
    Effect,
    EffectKind,
    Episode,
    EpisodeState,
    Observation,
    ObservationResult,
    Workflow,
    new_id,
)


@dataclass(frozen=True)
class Decision:
    """Result of folding one observation into an episode."""

    episode: Episode
    effect: Effect


def _fresh_candidate(episode: Episode, obs: Observation) -> Episode:
    """Begin a new candidate streak from ``obs``, minting a new episode id.

    A new id per candidate means the ``(workflow_id, episode_id)`` uniqueness
    constraint on events is what actually prevents duplicates, rather than
    relying on the dispatcher being careful.
    """
    return episode.model_copy(
        update={
            "id": new_id("ep"),
            "state": EpisodeState.CANDIDATE,
            "first_positive_monotonic": obs.captured_monotonic,
            "first_positive_at_utc": obs.captured_at_utc,
            "last_positive_monotonic": obs.captured_monotonic,
            "positive_samples": 1,
            "clear_streak": 0,
            "condition_unverified": False,
            "last_frame_id": obs.frame_id,
        }
    )


def _to_idle(episode: Episode, obs: Observation | None = None) -> Episode:
    """Reset to IDLE, clearing every pending timer."""
    return episode.model_copy(
        update={
            "state": EpisodeState.IDLE,
            "first_positive_monotonic": None,
            "first_positive_at_utc": None,
            "last_positive_monotonic": None,
            "positive_samples": 0,
            "clear_streak": 0,
            "condition_unverified": False,
            "last_frame_id": obs.frame_id if obs else episode.last_frame_id,
        }
    )


def _confirmed(episode: Episode, workflow: Workflow) -> bool:
    """True when both the duration and the sample count are satisfied.

    Both, never either: a burst of samples inside one second must not satisfy a
    ten-second rule, and one sample ten seconds after another must not either.
    """
    if episode.first_positive_monotonic is None:
        return False
    if episode.last_positive_monotonic is None:
        return False
    spanned = episode.last_positive_monotonic - episode.first_positive_monotonic
    return (
        spanned >= workflow.duration_seconds
        and episode.positive_samples >= workflow.min_samples
    )


def transition(episode: Episode, obs: Observation, workflow: Workflow) -> Decision:
    """Fold one observation into ``episode``.

    Returns the updated episode and the effect the caller must carry out.
    """
    # -- Guards ---------------------------------------------------------
    # Results from a superseded rule version cannot trigger the new rule.
    if obs.workflow_version != workflow.version:
        return Decision(
            episode,
            Effect(
                kind=EffectKind.REJECTED,
                episode_id=episode.id,
                reason=(
                    f"observation is for workflow version {obs.workflow_version}, "
                    f"current version is {workflow.version}"
                ),
            ),
        )

    if obs.workflow_id != workflow.id:
        return Decision(
            episode,
            Effect(
                kind=EffectKind.REJECTED,
                episode_id=episode.id,
                reason="observation belongs to a different workflow",
            ),
        )

    # Capture order, not arrival order. A frame we have already folded in, or
    # one older than it, is a duplicate or a late arrival; either way it must
    # not advance a timer a second time.
    if obs.frame_id <= episode.last_frame_id:
        return Decision(
            episode,
            Effect(
                kind=EffectKind.REJECTED,
                episode_id=episode.id,
                reason=(
                    f"stale or duplicate frame {obs.frame_id}; "
                    f"already processed through {episode.last_frame_id}"
                ),
            ),
        )

    state = episode.state
    result = obs.result

    # -- IDLE -----------------------------------------------------------
    if state == EpisodeState.IDLE:
        if result == ObservationResult.MET:
            candidate = _fresh_candidate(episode, obs)
            # A single sample can confirm only if the rule asks for one sample
            # and zero duration, which the schema forbids (duration_seconds>0).
            if _confirmed(candidate, workflow):
                return Decision(
                    candidate.model_copy(update={"state": EpisodeState.ACTIVE}),
                    Effect(
                        kind=EffectKind.TRIGGER_EVENT,
                        episode_id=candidate.id,
                        reason="confirmed on first sample",
                    ),
                )
            return Decision(candidate, Effect(episode_id=candidate.id))

        # not_met / unknown while idle: nothing to do but record the frame.
        return Decision(
            episode.model_copy(update={"last_frame_id": obs.frame_id}),
            Effect(episode_id=episode.id),
        )

    # -- CANDIDATE ------------------------------------------------------
    if state == EpisodeState.CANDIDATE:
        if result == ObservationResult.MET:
            last = episode.last_positive_monotonic
            gap = obs.captured_monotonic - last if last is not None else 0.0

            # A gap longer than max_gap_seconds breaks the streak. The guide is
            # explicit that the positive sample which arrives after a long gap
            # begins a fresh candidate rather than being discarded.
            if gap > workflow.max_gap_seconds:
                restarted = _fresh_candidate(episode, obs)
                return Decision(
                    restarted,
                    Effect(
                        episode_id=restarted.id,
                        reason=f"gap {gap:.2f}s exceeded max_gap_seconds; restarted",
                    ),
                )

            extended = episode.model_copy(
                update={
                    "last_positive_monotonic": obs.captured_monotonic,
                    "positive_samples": episode.positive_samples + 1,
                    "last_frame_id": obs.frame_id,
                }
            )
            if _confirmed(extended, workflow):
                return Decision(
                    extended.model_copy(
                        update={"state": EpisodeState.ACTIVE, "clear_streak": 0}
                    ),
                    Effect(
                        kind=EffectKind.TRIGGER_EVENT,
                        episode_id=extended.id,
                        reason=(
                            f"{extended.positive_samples} positive samples spanning "
                            f"{extended.last_positive_monotonic - extended.first_positive_monotonic:.2f}s"
                        ),
                    ),
                )
            return Decision(extended, Effect(episode_id=extended.id))

        # not_met or unknown: the condition is not continuously sustained, so
        # the candidate resets. Intermittent obstruction therefore cannot
        # complete a timer, which is the behaviour the acceptance set checks.
        reset = _to_idle(episode, obs)
        return Decision(
            reset,
            Effect(
                episode_id=reset.id,
                reason=f"candidate reset by {result.value}",
            ),
        )

    # -- ACTIVE ---------------------------------------------------------
    if state == EpisodeState.ACTIVE:
        if result == ObservationResult.MET:
            # Same episode continues. Explicitly no second event, no second
            # action: this is what stops a multi-minute condition producing a
            # stream of duplicate tasks.
            return Decision(
                episode.model_copy(
                    update={
                        "clear_streak": 0,
                        "condition_unverified": False,
                        "last_positive_monotonic": obs.captured_monotonic,
                        "last_frame_id": obs.frame_id,
                    }
                ),
                Effect(episode_id=episode.id, reason="episode continues"),
            )

        if result == ObservationResult.UNKNOWN:
            # Keep episode identity, mark unverified, reset the clear streak.
            # Critically: do not re-arm. An obscured lens is not evidence that
            # the condition cleared, so it must not close the episode either.
            return Decision(
                episode.model_copy(
                    update={
                        "condition_unverified": True,
                        "clear_streak": 0,
                        "last_frame_id": obs.frame_id,
                    }
                ),
                Effect(
                    episode_id=episode.id,
                    reason="unknown while active; condition unverified",
                ),
            )

        # not_met: count consecutive fresh clear results.
        streak = episode.clear_streak + 1
        if streak >= workflow.clear_samples:
            closed = _to_idle(episode, obs)
            return Decision(
                closed,
                Effect(
                    kind=EffectKind.CLOSE_EPISODE,
                    episode_id=episode.id,  # the id being closed, not the reset one
                    reason=f"{streak} consecutive clear samples",
                ),
            )
        return Decision(
            episode.model_copy(
                update={"clear_streak": streak, "last_frame_id": obs.frame_id}
            ),
            Effect(episode_id=episode.id, reason=f"clearing ({streak}/{workflow.clear_samples})"),
        )

    raise AssertionError(f"unhandled episode state: {state!r}")


def on_camera_unhealthy(episode: Episode) -> Decision:
    """Health watchdog: the camera went stale or disconnected.

    A pending candidate must not survive a gap it cannot account for, so its
    timer resets. An active episode keeps its identity and is held as
    unverified until clear evidence arrives -- losing the camera is not proof
    the condition ended.
    """
    if episode.state == EpisodeState.CANDIDATE:
        reset = _to_idle(episode)
        return Decision(
            reset,
            Effect(episode_id=reset.id, reason="candidate timer reset: camera unhealthy"),
        )

    if episode.state == EpisodeState.ACTIVE:
        return Decision(
            episode.model_copy(
                update={"condition_unverified": True, "clear_streak": 0}
            ),
            Effect(episode_id=episode.id, reason="active episode held: camera unhealthy"),
        )

    return Decision(episode, Effect(episode_id=episode.id))


def restore_after_restart(episode: Episode) -> Episode:
    """Rebuild in-memory state from a persisted episode after a restart.

    Active episode identity survives, so the same physical situation does not
    produce a second task on reboot. Pending timers do not survive, because the
    monotonic clock restarted and the frames that fed them are gone.
    """
    if episode.state == EpisodeState.ACTIVE:
        return episode.model_copy(
            update={
                "condition_unverified": True,
                "clear_streak": 0,
                "first_positive_monotonic": None,
                "last_positive_monotonic": None,
                "last_frame_id": -1,
            }
        )
    return _to_idle(episode).model_copy(update={"last_frame_id": -1})
