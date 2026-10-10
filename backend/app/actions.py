"""Actions: internal tasks and outbound webhooks.

Owner: C. A model verdict is not a completed workflow (guide step 7). An event
is only done when its action has a recorded outcome, and a failed delivery is
shown as failed rather than quietly dropped.

Delivery semantics, stated plainly: this is at-least-once. If a receiver
accepts a POST and the response is lost, the retry delivers it again. Every
payload therefore carries a stable ``idempotency_key`` derived from the event,
and the receiver is expected to deduplicate on it. We do not claim
exactly-once delivery to arbitrary endpoints.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

import httpx

from .db import Database
from .schemas import (
    ActionRecord,
    ActionStatus,
    ActionType,
    CreateTaskAction,
    Task,
    WebhookAction,
    new_id,
    utcnow,
)

log = logging.getLogger(__name__)

#: Retry backoff in seconds. Bounded on purpose: an endpoint that is down stays
#: down, and an unbounded queue of retries hides the failure from the operator.
DEFAULT_BACKOFF: tuple[float, ...] = (1.0, 2.0, 4.0)
DEFAULT_TIMEOUT_SECONDS = 5.0


def idempotency_key(event_id: str, action_type: ActionType) -> str:
    """Stable across retries and across restarts, because it is derived only
    from identifiers that do not change."""
    digest = hashlib.sha256(f"{event_id}:{action_type.value}".encode()).hexdigest()
    return digest[:32]


@dataclass
class WebhookDestination:
    id: str
    url: str
    label: str = ""
    #: Optional shared secret sent as a header so the receiver can authenticate.
    secret: Optional[str] = None


@dataclass
class DestinationAllowlist:
    """Backend-controlled set of webhook targets.

    The compiler selects a destination *id*; the URL is resolved here. Scene
    text, instructions and model output can therefore never cause a request to
    an arbitrary host.
    """

    destinations: dict[str, WebhookDestination] = field(default_factory=dict)

    def add(self, dest: WebhookDestination) -> None:
        self.destinations[dest.id] = dest

    def resolve(self, destination_id: str) -> WebhookDestination:
        try:
            return self.destinations[destination_id]
        except KeyError:
            raise ValueError(
                f"destination '{destination_id}' is not in the allowlist; "
                f"known: {sorted(self.destinations)}"
            ) from None

    def ids(self) -> list[str]:
        return sorted(self.destinations)


@dataclass
class DeliveryResult:
    ok: bool
    receipt: Optional[str] = None
    error: Optional[str] = None


class Dispatcher:
    """Reads pending action records and records their outcome.

    The dispatcher is separate from the engine so that a slow or failing
    endpoint cannot stall frame processing.
    """

    def __init__(
        self,
        db: Database,
        allowlist: DestinationAllowlist,
        *,
        backoff: tuple[float, ...] = DEFAULT_BACKOFF,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ) -> None:
        self.db = db
        self.allowlist = allowlist
        self.backoff = backoff
        self.timeout = timeout
        self._sleep = sleep  # injectable so tests do not actually wait
        self._transport = transport

    # -- task ------------------------------------------------------------

    def run_create_task(
        self, action: ActionRecord, spec: CreateTaskAction, *, workflow_id: str,
        evidence_path: Optional[str] = None,
    ) -> DeliveryResult:
        """Internal tasks are local writes, so they succeed or raise.

        ``UNIQUE(event_id)`` on tasks means a retry cannot create a second task
        for the same event; a duplicate attempt reports the existing one.
        """
        task = Task(
            title=spec.title,
            workflow_id=workflow_id,
            event_id=action.event_id,
            evidence_path=evidence_path,
        )
        created = self.db.create_task(task)
        if not created:
            return DeliveryResult(
                ok=True, receipt=f"task already exists for event {action.event_id}"
            )
        return DeliveryResult(ok=True, receipt=f"task:{task.id}")

    # -- webhook ----------------------------------------------------------

    def build_webhook_payload(
        self,
        action: ActionRecord,
        event: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "idempotency_key": action.idempotency_key,
            "event_id": event["id"],
            "workflow_id": event["workflow_id"],
            "event_type": "camera_condition_confirmed",
            "first_positive_at": event["first_positive_at"],
            "triggered_at": event["triggered_at"],
            "status": event["status"],
            "explanation": event["explanation"],
            "evidence": {
                "snapshot_path": event.get("snapshot_path"),
                "clip_path": event.get("clip_path"),
            },
        }

    async def run_webhook(
        self, action: ActionRecord, spec: WebhookAction, event: dict[str, Any]
    ) -> DeliveryResult:
        try:
            dest = self.allowlist.resolve(spec.destination_id)
        except ValueError as exc:
            # Not retryable: a missing destination is a configuration error.
            return DeliveryResult(ok=False, error=str(exc))

        payload = self.build_webhook_payload(action, event)
        headers = {
            "Content-Type": "application/json",
            "Idempotency-Key": action.idempotency_key,
        }
        if dest.secret:
            headers["X-Webhook-Secret"] = dest.secret

        last_error = "not attempted"
        async with httpx.AsyncClient(
            timeout=self.timeout, transport=self._transport
        ) as client:
            for attempt, delay in enumerate(self.backoff, start=1):
                try:
                    resp = await client.post(dest.url, json=payload, headers=headers)
                    if 200 <= resp.status_code < 300:
                        return DeliveryResult(
                            ok=True,
                            receipt=json.dumps(
                                {
                                    "status_code": resp.status_code,
                                    "destination": dest.id,
                                    "attempt": attempt,
                                    "body": resp.text[:500],
                                }
                            ),
                        )
                    last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
                    # 4xx other than 408/429 will not improve on retry.
                    if 400 <= resp.status_code < 500 and resp.status_code not in (408, 429):
                        return DeliveryResult(ok=False, error=last_error)
                except (httpx.TimeoutException, httpx.TransportError) as exc:
                    last_error = f"{type(exc).__name__}: {exc}"

                if attempt < len(self.backoff):
                    await self._sleep(delay)

        return DeliveryResult(ok=False, error=last_error)

    # -- driver -----------------------------------------------------------

    async def execute(
        self, action: ActionRecord, spec: CreateTaskAction | WebhookAction,
        *, workflow_id: str,
    ) -> ActionRecord:
        """Execute one action record and persist its outcome."""
        event = self.db.get_event(action.event_id)
        if event is None:
            action.status = ActionStatus.FAILED
            action.last_error = f"event {action.event_id} not found"
            action.attempts += 1
            self.db.update_action(action)
            return action

        if isinstance(spec, CreateTaskAction):
            result = self.run_create_task(
                action, spec, workflow_id=workflow_id,
                evidence_path=event.get("snapshot_path"),
            )
        else:
            result = await self.run_webhook(action, spec, event)

        action.attempts += 1
        action.updated_at = utcnow()
        if result.ok:
            action.status = ActionStatus.SUCCEEDED
            action.receipt = result.receipt
            action.last_error = None
        else:
            action.status = ActionStatus.FAILED
            action.last_error = result.error
        self.db.update_action(action)
        return action

    async def retry(
        self, action_id: str, spec: CreateTaskAction | WebhookAction, *, workflow_id: str
    ) -> Optional[ActionRecord]:
        """Operator-triggered retry of a failed delivery."""
        action = self.db.get_action(action_id)
        if action is None:
            return None
        if action.status == ActionStatus.SUCCEEDED:
            return action
        action.status = ActionStatus.PENDING
        self.db.update_action(action)
        return await self.execute(action, spec, workflow_id=workflow_id)


def build_action_record(event_id: str, action_type: ActionType) -> ActionRecord:
    return ActionRecord(
        id=new_id("act"),
        event_id=event_id,
        type=action_type,
        status=ActionStatus.PENDING,
        idempotency_key=idempotency_key(event_id, action_type),
    )
