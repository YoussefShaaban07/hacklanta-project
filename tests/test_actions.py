"""Action delivery tests: allowlist, retries, idempotency, honest failure.

No test sleeps. The dispatcher takes its sleep function by injection, so the
backoff schedule is asserted on rather than waited through.
"""

from __future__ import annotations

from datetime import timedelta

import httpx
import pytest

from app.actions import (
    DestinationAllowlist,
    Dispatcher,
    WebhookDestination,
    build_action_record,
    idempotency_key,
)
from app.db import Database
from app.schemas import (
    ActionStatus,
    ActionType,
    CreateTaskAction,
    Event,
    Task,
    WebhookAction,
)

from conftest import BASE_UTC, make_workflow


@pytest.fixture
def db(tmp_path) -> Database:
    database = Database(tmp_path / "actions.db")
    yield database
    database.close()


@pytest.fixture
def allowlist() -> DestinationAllowlist:
    al = DestinationAllowlist()
    al.add(WebhookDestination(id="test_receiver", url="https://receiver.test/hook"))
    return al


class RecordingSleep:
    """Captures the backoff schedule instead of waiting."""

    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, delay: float) -> None:
        self.delays.append(delay)


def seed_event(db: Database, workflow, episode_id="ep_1") -> Event:
    db.save_workflow(workflow)
    ev = Event(
        workflow_id=workflow.id,
        workflow_version=workflow.version,
        episode_id=episode_id,
        first_positive_at=BASE_UTC,
        triggered_at=BASE_UTC + timedelta(seconds=10),
        explanation="tray empty",
        snapshot_path="data/evidence/shot.jpg",
    )
    return ev


class TestIdempotencyKey:
    def test_stable_for_same_event_and_type(self):
        a = idempotency_key("evt_1", ActionType.WEBHOOK)
        b = idempotency_key("evt_1", ActionType.WEBHOOK)
        assert a == b, "a retry must reuse the key so the receiver can dedupe"

    def test_differs_across_events_and_types(self):
        assert idempotency_key("evt_1", ActionType.WEBHOOK) != idempotency_key(
            "evt_2", ActionType.WEBHOOK
        )
        assert idempotency_key("evt_1", ActionType.WEBHOOK) != idempotency_key(
            "evt_1", ActionType.CREATE_TASK
        )


class TestAllowlist:
    def test_resolves_known_destination(self, allowlist):
        assert allowlist.resolve("test_receiver").url == "https://receiver.test/hook"

    def test_rejects_unknown_destination(self, allowlist):
        with pytest.raises(ValueError, match="not in the allowlist"):
            allowlist.resolve("https://evil.example/exfil")

    async def test_unknown_destination_fails_without_network_call(self, db, allowlist):
        """A compiler that invented a destination must not cause a request."""
        workflow = make_workflow()
        ev = seed_event(db, workflow)
        action = build_action_record(ev.id, ActionType.WEBHOOK)
        db.commit_event_with_action(ev, action)

        def explode(request):  # pragma: no cover - must never run
            raise AssertionError(f"no request should be made, got {request.url}")

        dispatcher = Dispatcher(
            db, allowlist, transport=httpx.MockTransport(explode)
        )
        result = await dispatcher.run_webhook(
            action, WebhookAction(destination_id="not_allowed"), db.get_event(ev.id)
        )
        assert result.ok is False
        assert "not in the allowlist" in result.error


class TestCreateTask:
    async def test_creates_one_task_and_records_receipt(self, db, allowlist):
        workflow = make_workflow()
        ev = seed_event(db, workflow)
        action = build_action_record(ev.id, ActionType.CREATE_TASK)
        db.commit_event_with_action(ev, action)

        dispatcher = Dispatcher(db, allowlist)
        done = await dispatcher.execute(
            action, CreateTaskAction(title="Restock the supply tray"),
            workflow_id=workflow.id,
        )
        assert done.status == ActionStatus.SUCCEEDED
        assert done.receipt.startswith("task:")
        tasks = db.list_tasks()
        assert len(tasks) == 1
        assert tasks[0]["evidence_path"] == "data/evidence/shot.jpg"

    async def test_retry_does_not_create_a_second_task(self, db, allowlist):
        workflow = make_workflow()
        ev = seed_event(db, workflow)
        action = build_action_record(ev.id, ActionType.CREATE_TASK)
        db.commit_event_with_action(ev, action)
        dispatcher = Dispatcher(db, allowlist)
        spec = CreateTaskAction(title="Restock the supply tray")

        await dispatcher.execute(action, spec, workflow_id=workflow.id)
        await dispatcher.retry(action.id, spec, workflow_id=workflow.id)
        await dispatcher.retry(action.id, spec, workflow_id=workflow.id)

        assert len(db.list_tasks()) == 1


class TestWebhook:
    async def test_success_records_receipt(self, db, allowlist):
        workflow = make_workflow()
        ev = seed_event(db, workflow)
        action = build_action_record(ev.id, ActionType.WEBHOOK)
        db.commit_event_with_action(ev, action)

        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["key"] = request.headers.get("Idempotency-Key")
            seen["body"] = request.read().decode()
            return httpx.Response(200, text='{"ok":true}')

        dispatcher = Dispatcher(db, allowlist, transport=httpx.MockTransport(handler))
        done = await dispatcher.execute(
            action, WebhookAction(destination_id="test_receiver"),
            workflow_id=workflow.id,
        )
        assert done.status == ActionStatus.SUCCEEDED
        assert seen["key"] == action.idempotency_key
        assert ev.id in seen["body"]

    async def test_fails_then_recovers_on_retry(self, db, allowlist):
        """Guide scenario: receiver fails, then recovers; retry succeeds."""
        workflow = make_workflow()
        ev = seed_event(db, workflow)
        action = build_action_record(ev.id, ActionType.WEBHOOK)
        db.commit_event_with_action(ev, action)
        spec = WebhookAction(destination_id="test_receiver")

        state = {"up": False}
        keys_seen: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            keys_seen.append(request.headers["Idempotency-Key"])
            if not state["up"]:
                return httpx.Response(503, text="unavailable")
            return httpx.Response(200, text='{"ok":true}')

        sleeper = RecordingSleep()
        dispatcher = Dispatcher(
            db, allowlist, transport=httpx.MockTransport(handler), sleep=sleeper
        )

        done = await dispatcher.execute(action, spec, workflow_id=workflow.id)
        assert done.status == ActionStatus.FAILED
        assert "503" in done.last_error, "failure must be visible, not swallowed"

        state["up"] = True
        recovered = await dispatcher.retry(action.id, spec, workflow_id=workflow.id)
        assert recovered.status == ActionStatus.SUCCEEDED
        assert len(set(keys_seen)) == 1, "every attempt reuses one idempotency key"

    async def test_backoff_is_bounded_and_ordered(self, db, allowlist):
        workflow = make_workflow()
        ev = seed_event(db, workflow)
        action = build_action_record(ev.id, ActionType.WEBHOOK)
        db.commit_event_with_action(ev, action)

        attempts = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            attempts["n"] += 1
            return httpx.Response(500, text="boom")

        sleeper = RecordingSleep()
        dispatcher = Dispatcher(
            db, allowlist, transport=httpx.MockTransport(handler),
            backoff=(1.0, 2.0, 4.0), sleep=sleeper,
        )
        done = await dispatcher.execute(
            action, WebhookAction(destination_id="test_receiver"),
            workflow_id=workflow.id,
        )
        assert done.status == ActionStatus.FAILED
        assert attempts["n"] == 3, "bounded: three attempts, then stop"
        assert sleeper.delays == [1.0, 2.0], "sleeps between attempts, not after the last"

    async def test_client_error_is_not_retried(self, db, allowlist):
        """A 400 will not improve on retry; burning the budget hides the bug."""
        workflow = make_workflow()
        ev = seed_event(db, workflow)
        action = build_action_record(ev.id, ActionType.WEBHOOK)
        db.commit_event_with_action(ev, action)

        attempts = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            attempts["n"] += 1
            return httpx.Response(400, text="bad payload")

        dispatcher = Dispatcher(
            db, allowlist, transport=httpx.MockTransport(handler), sleep=RecordingSleep()
        )
        done = await dispatcher.execute(
            action, WebhookAction(destination_id="test_receiver"),
            workflow_id=workflow.id,
        )
        assert done.status == ActionStatus.FAILED
        assert attempts["n"] == 1

    async def test_timeout_is_retried(self, db, allowlist):
        workflow = make_workflow()
        ev = seed_event(db, workflow)
        action = build_action_record(ev.id, ActionType.WEBHOOK)
        db.commit_event_with_action(ev, action)

        attempts = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            attempts["n"] += 1
            raise httpx.ConnectTimeout("timed out")

        dispatcher = Dispatcher(
            db, allowlist, transport=httpx.MockTransport(handler), sleep=RecordingSleep()
        )
        done = await dispatcher.execute(
            action, WebhookAction(destination_id="test_receiver"),
            workflow_id=workflow.id,
        )
        assert done.status == ActionStatus.FAILED
        assert attempts["n"] == 3
        assert "Timeout" in done.last_error

    async def test_payload_carries_evidence_and_timestamps(self, db, allowlist):
        workflow = make_workflow()
        ev = seed_event(db, workflow)
        action = build_action_record(ev.id, ActionType.WEBHOOK)
        db.commit_event_with_action(ev, action)

        dispatcher = Dispatcher(db, allowlist)
        payload = dispatcher.build_webhook_payload(action, db.get_event(ev.id))

        assert payload["event_id"] == ev.id
        assert payload["workflow_id"] == workflow.id
        assert payload["evidence"]["snapshot_path"] == "data/evidence/shot.jpg"
        assert payload["idempotency_key"] == action.idempotency_key
        assert payload["first_positive_at"] and payload["triggered_at"]
