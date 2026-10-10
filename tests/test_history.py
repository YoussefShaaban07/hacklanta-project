"""History query tests.

Two properties carry the weight here: answers are computed from the database
(never from a model), and absence of records is reported as absence of
*records*, not as a claim that nothing happened.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.actions import build_action_record
from app.db import Database
from app.history import answer_question
from app.schemas import ActionStatus, ActionType, Event

from conftest import make_workflow

NOW = datetime(2026, 10, 9, 15, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def db(tmp_path):
    database = Database(tmp_path / "history.db")
    workflow = make_workflow(id="wf_h")
    database.save_workflow(workflow)
    yield database
    database.close()


def add_event(db: Database, when: datetime, episode: str, failed: bool = False) -> Event:
    event = Event(
        workflow_id="wf_h", episode_id=episode,
        first_positive_at=when - timedelta(seconds=10),
        triggered_at=when, explanation="tray empty",
    )
    action = build_action_record(event.id, ActionType.WEBHOOK)
    db.commit_event_with_action(event, action)
    if failed:
        action.status = ActionStatus.FAILED
        action.last_error = "HTTP 503"
        db.update_action(action)
    return event


class TestCounts:
    def test_counts_today(self, db):
        for i in range(3):
            add_event(db, NOW - timedelta(hours=i), f"ep_{i}")
        result = answer_question(db, "How many events occurred today?",
                                 workflow_id="wf_h", now=NOW)
        assert result.supported
        assert "3" in result.answer

    def test_count_excludes_other_days(self, db):
        add_event(db, NOW, "ep_today")
        add_event(db, NOW - timedelta(days=3), "ep_old")
        result = answer_question(db, "How many events occurred today?",
                                 workflow_id="wf_h", now=NOW)
        assert "1" in result.answer

    def test_zero_is_a_real_answer(self, db):
        result = answer_question(db, "How many events occurred today?",
                                 workflow_id="wf_h", now=NOW)
        assert result.supported
        assert "0" in result.answer

    def test_every_answer_states_its_scope(self, db):
        add_event(db, NOW, "ep_1")
        result = answer_question(db, "How many events occurred today?",
                                 workflow_id="wf_h", now=NOW)
        assert "recorded observations" in result.scope_note


class TestLastTrigger:
    def test_reports_the_latest(self, db):
        add_event(db, NOW - timedelta(hours=2), "ep_old")
        add_event(db, NOW - timedelta(minutes=5), "ep_new")
        result = answer_question(db, "When did this rule last trigger?",
                                 workflow_id="wf_h", now=NOW)
        assert result.supported
        assert result.events[0]["episode_id"] == "ep_new"

    def test_no_records_is_not_nothing_happened(self, db):
        result = answer_question(db, "When did this rule last trigger?",
                                 workflow_id="wf_h", now=NOW)
        assert result.supported
        assert "no matching recorded events" in result.answer.lower()
        # Explicitly NOT a claim about the world.
        assert "nothing happened" not in result.answer.lower()


class TestFailedActions:
    def test_lists_failures(self, db):
        add_event(db, NOW, "ep_ok")
        add_event(db, NOW - timedelta(minutes=1), "ep_bad", failed=True)
        result = answer_question(db, "Show failed actions.", now=NOW)
        assert result.supported
        assert len(result.actions) == 1
        assert result.actions[0]["last_error"] == "HTTP 503"

    def test_none_failed(self, db):
        add_event(db, NOW, "ep_ok")
        result = answer_question(db, "Show failed actions.", now=NOW)
        assert result.supported
        assert "no failed actions" in result.answer.lower()


class TestWindows:
    def test_last_n_hours(self, db):
        add_event(db, NOW - timedelta(minutes=30), "ep_recent")
        add_event(db, NOW - timedelta(hours=5), "ep_old")
        result = answer_question(db, "What happened in the last 2 hours?",
                                 workflow_id="wf_h", now=NOW)
        assert result.supported
        assert len(result.events) == 1


class TestUnsupported:
    @pytest.mark.parametrize("question", [
        "Who was in the warehouse yesterday?",
        "Will the tray run out tomorrow?",
        "Delete all the events",
    ])
    def test_explains_instead_of_guessing(self, db, question):
        result = answer_question(db, question, now=NOW)
        assert not result.supported
        assert result.suggestions

    def test_empty_question(self, db):
        result = answer_question(db, "", now=NOW)
        assert not result.supported

    def test_no_sql_is_accepted_from_the_question(self, db):
        """A question containing SQL is just unmatched text, never executed."""
        add_event(db, NOW, "ep_1")
        result = answer_question(db, "'; DROP TABLE events; --", now=NOW)
        assert not result.supported
        # The table is still there and still has its row.
        assert db.count_events(workflow_id="wf_h") == 1
