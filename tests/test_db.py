"""Storage tests: duplicate prevention, atomic commit, restart survival."""

from __future__ import annotations

from datetime import timedelta

import pytest

from app.actions import build_action_record
from app.db import Database
from app.engine import restore_after_restart, transition
from app.schemas import (
    ActionType,
    EffectKind,
    Episode,
    EpisodeState,
    Event,
    EventStatus,
    ObservationResult,
    Task,
    utcnow,
)

from conftest import BASE_UTC, Clock, make_workflow

MET = ObservationResult.MET
NOT_MET = ObservationResult.NOT_MET


@pytest.fixture
def db(tmp_path) -> Database:
    database = Database(tmp_path / "test.db")
    yield database
    database.close()


def make_event(workflow, episode_id: str, **overrides) -> Event:
    defaults = dict(
        workflow_id=workflow.id,
        workflow_version=workflow.version,
        episode_id=episode_id,
        first_positive_at=BASE_UTC,
        triggered_at=BASE_UTC + timedelta(seconds=10),
        explanation="tray empty across 3 samples",
    )
    defaults.update(overrides)
    return Event(**defaults)


class TestEventCommit:
    def test_event_and_action_commit_together(self, db, workflow):
        db.save_workflow(workflow)
        event = make_event(workflow, "ep_1")
        action = build_action_record(event.id, ActionType.CREATE_TASK)

        assert db.commit_event_with_action(event, action) is True

        assert db.get_event(event.id) is not None
        actions = db.list_actions(event.id)
        assert len(actions) == 1
        assert actions[0].status.value == "pending"

    def test_same_episode_cannot_produce_two_events(self, db, workflow):
        """The uniqueness constraint, not dispatcher care, prevents duplicates."""
        db.save_workflow(workflow)
        first = make_event(workflow, "ep_shared")
        second = make_event(workflow, "ep_shared")

        assert db.commit_event_with_action(
            first, build_action_record(first.id, ActionType.CREATE_TASK)
        ) is True
        assert db.commit_event_with_action(
            second, build_action_record(second.id, ActionType.CREATE_TASK)
        ) is False

        assert len(db.list_events(workflow_id=workflow.id)) == 1

    def test_different_episodes_do_produce_two_events(self, db, workflow):
        db.save_workflow(workflow)
        for episode_id in ("ep_1", "ep_2"):
            ev = make_event(workflow, episode_id)
            assert db.commit_event_with_action(
                ev, build_action_record(ev.id, ActionType.CREATE_TASK)
            ) is True
        assert len(db.list_events(workflow_id=workflow.id)) == 2

    def test_task_is_unique_per_event(self, db, workflow):
        db.save_workflow(workflow)
        ev = make_event(workflow, "ep_1")
        db.commit_event_with_action(ev, build_action_record(ev.id, ActionType.CREATE_TASK))

        t1 = Task(title="Restock", workflow_id=workflow.id, event_id=ev.id)
        t2 = Task(title="Restock again", workflow_id=workflow.id, event_id=ev.id)
        assert db.create_task(t1) is True
        assert db.create_task(t2) is False, "retry must not create a second task"
        assert len(db.list_tasks()) == 1

    def test_evidence_attaches_after_the_fact(self, db, workflow):
        """Snapshot is available immediately; the clip lands later."""
        db.save_workflow(workflow)
        ev = make_event(workflow, "ep_1", snapshot_path="data/evidence/a.jpg")
        db.commit_event_with_action(ev, build_action_record(ev.id, ActionType.CREATE_TASK))

        assert db.get_event(ev.id)["clip_path"] is None
        db.attach_evidence(ev.id, clip_path="data/evidence/a.mp4")
        stored = db.get_event(ev.id)
        assert stored["clip_path"] == "data/evidence/a.mp4"
        assert stored["snapshot_path"] == "data/evidence/a.jpg"


class TestQueries:
    def test_filters_by_workflow_and_time(self, db):
        wf_a, wf_b = make_workflow(id="wf_a"), make_workflow(id="wf_b")
        db.save_workflow(wf_a)
        db.save_workflow(wf_b)
        for i in range(3):
            ev = make_event(wf_a, f"ep_a{i}", triggered_at=BASE_UTC + timedelta(hours=i))
            db.commit_event_with_action(ev, build_action_record(ev.id, ActionType.CREATE_TASK))
        ev_b = make_event(wf_b, "ep_b0")
        db.commit_event_with_action(ev_b, build_action_record(ev_b.id, ActionType.CREATE_TASK))

        assert len(db.list_events(workflow_id="wf_a")) == 3
        assert len(db.list_events(workflow_id="wf_b")) == 1
        assert db.count_events(workflow_id="wf_a") == 3
        assert len(db.list_events(since=BASE_UTC + timedelta(hours=1))) == 2

    def test_count_distinguishes_zero_from_missing(self, db, workflow):
        """'No matching recorded events' is a real answer, not an error."""
        db.save_workflow(workflow)
        assert db.count_events(workflow_id=workflow.id) == 0
        assert db.list_events(workflow_id=workflow.id) == []


class TestRestart:
    def test_active_episode_survives_restart_without_refiring(self, tmp_path, workflow):
        """The full guide scenario: restart while an event is active."""
        path = tmp_path / "restart.db"
        db = Database(path)
        db.save_workflow(workflow)
        clock = Clock(workflow)

        # Drive to ACTIVE and commit the event.
        ep = Episode(workflow_id=workflow.id, workflow_version=workflow.version)
        for t in (0, 5, 10):
            decision = transition(ep, clock.obs(MET, t), workflow)
            ep = decision.episode
            if decision.effect.kind == EffectKind.TRIGGER_EVENT:
                ev = make_event(workflow, decision.effect.episode_id)
                db.commit_event_with_action(
                    ev, build_action_record(ev.id, ActionType.CREATE_TASK),
                    task=Task(title="Restock", workflow_id=workflow.id, event_id=ev.id),
                )
        assert ep.state == EpisodeState.ACTIVE
        db.save_episode(ep)
        active_episode_id = ep.id
        db.close()

        # --- restart ---
        db2 = Database(path)
        reloaded = db2.load_episodes(workflow.id)[0]
        restored = restore_after_restart(reloaded)
        assert restored.id == active_episode_id
        assert restored.state == EpisodeState.ACTIVE

        # History retained.
        assert len(db2.list_events(workflow_id=workflow.id)) == 1
        assert len(db2.list_tasks()) == 1

        # The condition is still true after reboot. It must not make a new task.
        fresh = Clock(workflow)
        for t in (0, 5, 10, 15, 20):
            decision = transition(restored, fresh.obs(MET, t), workflow)
            restored = decision.episode
            assert decision.effect.kind != EffectKind.TRIGGER_EVENT

        assert len(db2.list_events(workflow_id=workflow.id)) == 1
        assert len(db2.list_tasks()) == 1, "no repeated task for the same episode"
        db2.close()

    def test_workflow_versions_are_retained(self, db):
        v1 = make_workflow(version=1, duration_seconds=10)
        v2 = make_workflow(version=2, duration_seconds=25)
        db.save_workflow(v1)
        db.save_workflow(v2)
        assert db.get_workflow(v1.id).version == 2, "latest wins by default"
        assert db.get_workflow(v1.id, version=1).duration_seconds == 10
        assert len(db.list_workflows()) == 1, "list shows one row per workflow id"
