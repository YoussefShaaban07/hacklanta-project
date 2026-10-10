"""State machine tests.

Organised in two layers:
  * ``TestStateTable``  - one test per row of the guide's step-6 table.
  * ``TestAcceptanceScenarios`` - the step-11 scenario table, which is what the
    demo is actually judged on.

Everything runs on synthetic timestamped observations: no camera, no model, no
sleeping.
"""

from __future__ import annotations

import pytest

from app.engine import (
    Decision,
    on_camera_unhealthy,
    restore_after_restart,
    transition,
)
from app.schemas import (
    EffectKind,
    Episode,
    EpisodeState,
    ObservationResult,
)

from conftest import Clock, make_workflow

MET = ObservationResult.MET
NOT_MET = ObservationResult.NOT_MET
UNKNOWN = ObservationResult.UNKNOWN


def drive(episode: Episode, clock: Clock, workflow, script) -> tuple[Episode, list]:
    """Feed ``(result, offset)`` pairs through the engine.

    Returns the final episode and every effect produced, so a test can assert
    both the end state and that exactly one trigger happened along the way.
    """
    effects = []
    for result, at in script:
        decision: Decision = transition(episode, clock.obs(result, at), workflow)
        episode = decision.episode
        effects.append(decision.effect)
    return episode, effects


def triggers(effects) -> list:
    return [e for e in effects if e.kind == EffectKind.TRIGGER_EVENT]


# ==========================================================================
# State table (guide step 6)
# ==========================================================================


class TestStateTable:
    def test_idle_plus_met_starts_candidate(self, episode, clock, workflow):
        d = transition(episode, clock.obs(MET, 0), workflow)
        assert d.episode.state == EpisodeState.CANDIDATE
        assert d.episode.positive_samples == 1
        assert d.episode.first_positive_monotonic is not None
        assert d.effect.kind == EffectKind.NONE

    def test_idle_plus_not_met_stays_idle(self, episode, clock, workflow):
        d = transition(episode, clock.obs(NOT_MET, 0), workflow)
        assert d.episode.state == EpisodeState.IDLE
        assert d.episode.positive_samples == 0

    def test_idle_plus_unknown_stays_idle(self, episode, clock, workflow):
        d = transition(episode, clock.obs(UNKNOWN, 0), workflow)
        assert d.episode.state == EpisodeState.IDLE

    def test_candidate_extends_streak_within_gap(self, episode, clock, workflow):
        ep, effects = drive(episode, clock, workflow, [(MET, 0), (MET, 2), (MET, 4)])
        assert ep.state == EpisodeState.CANDIDATE  # 4s < 10s duration
        assert ep.positive_samples == 3
        assert triggers(effects) == []

    def test_trigger_requires_both_duration_and_samples(self, episode, clock, workflow):
        """Duration satisfied but too few samples: must not trigger.

        Samples are spaced 4s apart to stay inside max_gap_seconds=5, so the
        streak survives and this isolates the min_samples condition alone.
        """
        wf = make_workflow(min_samples=5)
        ep, effects = drive(episode, clock, wf, [(MET, 0), (MET, 4), (MET, 8), (MET, 12)])
        assert triggers(effects) == []
        assert ep.state == EpisodeState.CANDIDATE
        assert ep.positive_samples == 4  # spans 12s >= 10s, but 4 < min_samples=5

    def test_trigger_requires_duration_not_just_samples(self, episode, clock, workflow):
        """Ten samples inside one second must not satisfy a ten-second rule."""
        script = [(MET, i * 0.1) for i in range(10)]
        ep, effects = drive(episode, clock, workflow, script)
        assert triggers(effects) == []
        assert ep.state == EpisodeState.CANDIDATE

    def test_candidate_triggers_when_both_pass(self, episode, clock, workflow):
        ep, effects = drive(
            episode, clock, workflow, [(MET, 0), (MET, 5), (MET, 10)]
        )
        assert ep.state == EpisodeState.ACTIVE
        assert len(triggers(effects)) == 1

    @pytest.mark.parametrize("breaker", [NOT_MET, UNKNOWN])
    def test_candidate_reset_by_negative_or_unknown(self, episode, clock, workflow, breaker):
        ep, effects = drive(
            episode, clock, workflow, [(MET, 0), (MET, 4), (breaker, 6), (MET, 8)]
        )
        # The breaker reset the streak; the later MET started a new candidate
        # whose timer begins at t=8, so nothing has triggered.
        assert triggers(effects) == []
        assert ep.state == EpisodeState.CANDIDATE
        assert ep.positive_samples == 1

    def test_long_gap_restarts_candidate_rather_than_discarding(self, episode, clock, workflow):
        """A positive after a long gap begins a fresh candidate."""
        ep, effects = drive(episode, clock, workflow, [(MET, 0), (MET, 30)])
        assert ep.state == EpisodeState.CANDIDATE
        assert ep.positive_samples == 1
        assert ep.first_positive_monotonic == clock.obs(MET, 30).captured_monotonic
        assert triggers(effects) == []

    def test_active_plus_met_does_not_retrigger(self, episode, clock, workflow):
        script = [(MET, 0), (MET, 5), (MET, 10)] + [(MET, 10 + i) for i in range(1, 30)]
        ep, effects = drive(episode, clock, workflow, script)
        assert ep.state == EpisodeState.ACTIVE
        assert len(triggers(effects)) == 1, "a sustained condition must fire once"

    def test_active_plus_unknown_holds_episode_unverified(self, episode, clock, workflow):
        ep, effects = drive(
            episode, clock, workflow,
            [(MET, 0), (MET, 5), (MET, 10), (UNKNOWN, 12)],
        )
        assert ep.state == EpisodeState.ACTIVE, "unknown must not close the episode"
        assert ep.condition_unverified is True
        assert len(triggers(effects)) == 1

    def test_unknown_resets_clear_streak_so_it_cannot_rearm(self, episode, clock, workflow):
        """not_met, not_met, unknown, not_met must not reach clear_samples=3."""
        ep, _ = drive(
            episode, clock, workflow,
            [(MET, 0), (MET, 5), (MET, 10),
             (NOT_MET, 12), (NOT_MET, 14), (UNKNOWN, 16), (NOT_MET, 18)],
        )
        assert ep.state == EpisodeState.ACTIVE
        assert ep.clear_streak == 1

    def test_active_closes_after_clear_samples(self, episode, clock, workflow):
        ep, effects = drive(
            episode, clock, workflow,
            [(MET, 0), (MET, 5), (MET, 10), (NOT_MET, 12), (NOT_MET, 14), (NOT_MET, 16)],
        )
        assert ep.state == EpisodeState.IDLE
        assert any(e.kind == EffectKind.CLOSE_EPISODE for e in effects)


# ==========================================================================
# Ordering, duplication and versioning guards
# ==========================================================================


class TestGuards:
    def test_duplicate_frame_is_rejected(self, episode, clock, workflow):
        d1 = transition(episode, clock.obs(MET, 0, frame_id=7), workflow)
        d2 = transition(d1.episode, clock.obs(MET, 2, frame_id=7), workflow)
        assert d2.effect.kind == EffectKind.REJECTED
        assert d2.episode.positive_samples == 1, "a replayed frame must not count twice"

    def test_out_of_order_frame_is_rejected(self, episode, clock, workflow):
        d1 = transition(episode, clock.obs(MET, 10, frame_id=10), workflow)
        d2 = transition(d1.episode, clock.obs(MET, 4, frame_id=4), workflow)
        assert d2.effect.kind == EffectKind.REJECTED
        assert d2.episode.positive_samples == 1

    def test_superseded_workflow_version_cannot_trigger(self, episode, clock, workflow):
        """Editing a rule mid-inference: the in-flight result is for v1, the
        rule is now v2, so it must be discarded."""
        wf_v2 = make_workflow(version=2)
        ep = Episode(workflow_id=wf_v2.id, workflow_version=2)
        d = transition(ep, clock.obs(MET, 0, version=1), wf_v2)
        assert d.effect.kind == EffectKind.REJECTED
        assert "version" in d.effect.reason
        assert d.episode.state == EpisodeState.IDLE

    def test_observation_for_another_workflow_is_rejected(self, episode, clock, workflow):
        other = make_workflow(id="wf_other")
        d = transition(episode, clock.obs(MET, 0), other)
        assert d.effect.kind == EffectKind.REJECTED


# ==========================================================================
# Camera health and restart
# ==========================================================================


class TestHealthAndRestart:
    def test_watchdog_resets_pending_candidate(self, episode, clock, workflow):
        ep, _ = drive(episode, clock, workflow, [(MET, 0), (MET, 4)])
        assert ep.state == EpisodeState.CANDIDATE
        d = on_camera_unhealthy(ep)
        assert d.episode.state == EpisodeState.IDLE
        assert d.episode.first_positive_monotonic is None

    def test_watchdog_holds_active_episode(self, episode, clock, workflow):
        ep, _ = drive(episode, clock, workflow, [(MET, 0), (MET, 5), (MET, 10)])
        d = on_camera_unhealthy(ep)
        assert d.episode.state == EpisodeState.ACTIVE
        assert d.episode.condition_unverified is True

    def test_restart_keeps_active_identity_but_drops_timers(self, episode, clock, workflow):
        ep, _ = drive(episode, clock, workflow, [(MET, 0), (MET, 5), (MET, 10)])
        original_id = ep.id
        restored = restore_after_restart(ep)
        assert restored.id == original_id, "same episode must not re-fire after reboot"
        assert restored.state == EpisodeState.ACTIVE
        assert restored.first_positive_monotonic is None
        assert restored.last_frame_id == -1

    def test_restart_discards_pending_candidate(self, episode, clock, workflow):
        ep, _ = drive(episode, clock, workflow, [(MET, 0), (MET, 4)])
        restored = restore_after_restart(ep)
        assert restored.state == EpisodeState.IDLE
        assert restored.positive_samples == 0

    def test_frame_ids_restart_from_zero_after_reboot(self, episode, clock, workflow):
        """A new capture worker numbers frames from 0 again; the restored
        episode must not reject them all as stale."""
        ep, _ = drive(episode, clock, workflow, [(MET, 0), (MET, 5), (MET, 10)])
        restored = restore_after_restart(ep)
        fresh = Clock(workflow)
        d = transition(restored, fresh.obs(NOT_MET, 0), workflow)
        assert d.effect.kind != EffectKind.REJECTED


# ==========================================================================
# Acceptance scenarios (guide step 11)
# ==========================================================================


class TestAcceptanceScenarios:
    def test_condition_longer_than_duration_yields_one_event(self, episode, clock, workflow):
        ep, effects = drive(
            episode, clock, workflow, [(MET, t) for t in range(0, 20, 2)]
        )
        assert len(triggers(effects)) == 1
        assert ep.state == EpisodeState.ACTIVE

    def test_brief_condition_yields_no_event(self, episode, clock, workflow):
        _, effects = drive(
            episode, clock, workflow, [(MET, 0), (MET, 2), (NOT_MET, 4)]
        )
        assert triggers(effects) == []

    def test_multi_minute_condition_yields_no_duplicates(self, episode, clock, workflow):
        script = [(MET, t) for t in range(0, 300, 2)]
        ep, effects = drive(episode, clock, workflow, script)
        assert len(triggers(effects)) == 1
        assert ep.state == EpisodeState.ACTIVE

    def test_clear_then_recur_yields_a_second_event(self, episode, clock, workflow):
        script = (
            [(MET, 0), (MET, 5), (MET, 10)]              # event 1
            + [(NOT_MET, 12), (NOT_MET, 14), (NOT_MET, 16)]  # close
            + [(MET, 20), (MET, 25), (MET, 30)]          # event 2
        )
        ep, effects = drive(episode, clock, workflow, script)
        fired = triggers(effects)
        assert len(fired) == 2, "a genuinely re-occurring condition must re-arm"
        assert fired[0].episode_id != fired[1].episode_id
        assert ep.state == EpisodeState.ACTIVE

    def test_occlusion_cannot_complete_the_timer(self, episode, clock, workflow):
        """Lens covered intermittently throughout: never a confirmed event."""
        script = []
        for i in range(30):
            script.append((MET, i * 2))
            script.append((UNKNOWN, i * 2 + 1))
        _, effects = drive(episode, clock, workflow, script)
        assert triggers(effects) == [], "unknown must break the streak every time"

    def test_camera_disconnect_during_candidate_resets_timer(self, episode, clock, workflow):
        ep, _ = drive(episode, clock, workflow, [(MET, 0), (MET, 4), (MET, 8)])
        assert ep.state == EpisodeState.CANDIDATE
        ep = on_camera_unhealthy(ep).episode
        # Reconnects and the condition is still present, but the clock restarts.
        ep2, effects = drive(ep, clock, workflow, [(MET, 20), (MET, 22)])
        assert triggers(effects) == []
        assert ep2.positive_samples == 2

    def test_api_failure_is_unknown_and_does_not_confirm(self, episode, clock, workflow):
        """Timeouts and malformed JSON arrive as unknown, which cannot trigger."""
        script = [(UNKNOWN, t) for t in range(0, 40, 2)]
        ep, effects = drive(episode, clock, workflow, script)
        assert triggers(effects) == []
        assert ep.state == EpisodeState.IDLE

    def test_edit_during_inference_cannot_trigger_new_rule(self, episode, clock, workflow):
        """Old-version results arriving after an edit are all rejected."""
        wf_v2 = make_workflow(version=2, duration_seconds=10.0)
        ep = Episode(workflow_id=wf_v2.id, workflow_version=2)
        script_effects = []
        for t in (0, 5, 10, 15):
            d = transition(ep, clock.obs(MET, t, version=1), wf_v2)
            ep = d.episode
            script_effects.append(d.effect)
        assert all(e.kind == EffectKind.REJECTED for e in script_effects)
        assert ep.state == EpisodeState.IDLE


# ==========================================================================
# Properties
# ==========================================================================


class TestProperties:
    def test_sustained_condition_fires_exactly_once_at_any_sample_rate(self, workflow):
        """Whatever the sampling interval, a condition held for five minutes
        produces one event, never two."""
        for interval in (0.5, 1.0, 2.0, 3.0, 4.9):
            clock = Clock(workflow)
            ep = Episode(workflow_id=workflow.id, workflow_version=workflow.version)
            n = int(300 / interval)
            script = [(MET, i * interval) for i in range(n)]
            ep, effects = drive(ep, clock, workflow, script)
            assert len(triggers(effects)) == 1, f"interval={interval}"

    def test_interval_above_max_gap_never_confirms(self, workflow):
        """Sampling slower than max_gap_seconds can never build a streak, which
        is a real misconfiguration the team should see in testing."""
        clock = Clock(workflow)
        ep = Episode(workflow_id=workflow.id, workflow_version=workflow.version)
        script = [(MET, i * 6.0) for i in range(20)]  # 6s > max_gap 5s
        ep, effects = drive(ep, clock, workflow, script)
        assert triggers(effects) == []
        assert ep.positive_samples == 1

    def test_episode_id_changes_between_confirmations(self, episode, clock, workflow):
        script = (
            [(MET, 0), (MET, 5), (MET, 10)]
            + [(NOT_MET, 12), (NOT_MET, 14), (NOT_MET, 16)]
            + [(MET, 20), (MET, 25), (MET, 30)]
            + [(NOT_MET, 32), (NOT_MET, 34), (NOT_MET, 36)]
            + [(MET, 40), (MET, 45), (MET, 50)]
        )
        _, effects = drive(episode, clock, workflow, script)
        ids = [e.episode_id for e in triggers(effects)]
        assert len(ids) == 3
        assert len(set(ids)) == 3, "each confirmation needs its own dedup identity"
