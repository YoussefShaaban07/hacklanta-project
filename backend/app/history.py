"""Answering questions from stored evidence.

Owner: C, with B. Questions are matched to a small set of supported shapes and
turned into *parameterized* queries. The model never writes SQL and never sees
the database; at most it could rephrase a summary of rows already returned.

The language matters here. A query that finds nothing answers "no matching
recorded events", never "nothing happened" -- the system only knows what it
sampled, and saying otherwise would claim knowledge of unobserved time.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta, timezone
from typing import Any, Optional

from .db import Database
from .schemas import ActionStatus


@dataclass
class HistoryAnswer:
    supported: bool
    kind: str
    answer: str
    events: list[dict[str, Any]] = field(default_factory=list)
    actions: list[dict[str, Any]] = field(default_factory=list)
    #: Always present on a successful answer: what the number actually covers.
    scope_note: str = ""
    suggestions: list[str] = field(default_factory=list)


SUPPORTED_EXAMPLES = [
    "When did this rule last trigger?",
    "How many events occurred today?",
    "Show failed actions.",
    "What happened in the last hour?",
]

_SCOPE_NOTE = (
    "Covers recorded observations only, not every instant between samples."
)


def _today_start(now: datetime) -> datetime:
    return datetime.combine(now.date(), time.min, tzinfo=timezone.utc)


def _fmt(dt_str: str) -> str:
    try:
        return datetime.fromisoformat(dt_str).strftime("%Y-%m-%d %H:%M:%S UTC")
    except (ValueError, TypeError):
        return dt_str


def answer_question(
    db: Database,
    question: str,
    *,
    workflow_id: Optional[str] = None,
    now: Optional[datetime] = None,
) -> HistoryAnswer:
    """Map a question onto a supported, parameterized query."""
    now = now or datetime.now(timezone.utc)
    q = (question or "").strip().lower()

    if not q:
        return HistoryAnswer(
            supported=False, kind="empty",
            answer="Ask a question about recorded events.",
            suggestions=SUPPORTED_EXAMPLES,
        )

    # -- failed actions -------------------------------------------------
    if "failed" in q and ("action" in q or "delivery" in q or "webhook" in q):
        actions = [
            a.model_dump(mode="json")
            for a in db.list_actions()
            if a.status == ActionStatus.FAILED
        ]
        if not actions:
            return HistoryAnswer(
                supported=True, kind="failed_actions",
                answer="No failed actions are recorded.",
                scope_note=_SCOPE_NOTE,
            )
        return HistoryAnswer(
            supported=True, kind="failed_actions",
            answer=f"{len(actions)} failed action(s) recorded.",
            actions=actions, scope_note=_SCOPE_NOTE,
        )

    # -- last trigger ---------------------------------------------------
    if ("last" in q or "most recent" in q or "when did" in q) and "trigger" in q \
            or ("when" in q and "last" in q):
        events = db.list_events(workflow_id=workflow_id, limit=1)
        if not events:
            return HistoryAnswer(
                supported=True, kind="last_trigger",
                answer="No matching recorded events.", scope_note=_SCOPE_NOTE,
            )
        return HistoryAnswer(
            supported=True, kind="last_trigger",
            answer=f"Last triggered at {_fmt(events[0]['triggered_at'])}.",
            events=events, scope_note=_SCOPE_NOTE,
        )

    # -- counts ---------------------------------------------------------
    if "how many" in q or "count" in q:
        if "today" in q:
            since, window = _today_start(now), "today"
        elif "hour" in q:
            since, window = now - timedelta(hours=1), "in the last hour"
        elif "week" in q:
            since, window = now - timedelta(days=7), "in the last 7 days"
        else:
            since, window = None, "in total"
        count = db.count_events(workflow_id=workflow_id, since=since)
        events = db.list_events(workflow_id=workflow_id, since=since, limit=50)
        noun = "event" if count == 1 else "events"
        return HistoryAnswer(
            supported=True, kind="count",
            answer=f"{count} recorded {noun} {window}.",
            events=events, scope_note=_SCOPE_NOTE,
        )

    # -- time window ----------------------------------------------------
    window_match = re.search(r"last\s+(\d+)\s*(minute|hour|day)s?", q)
    if window_match or "what happened" in q or "show" in q:
        if window_match:
            n, unit = int(window_match.group(1)), window_match.group(2)
            delta = {"minute": timedelta(minutes=n), "hour": timedelta(hours=n),
                     "day": timedelta(days=n)}[unit]
            since, window = now - delta, f"in the last {n} {unit}(s)"
        elif "today" in q:
            since, window = _today_start(now), "today"
        else:
            since, window = None, "in total"
        events = db.list_events(workflow_id=workflow_id, since=since, limit=50)
        if not events:
            return HistoryAnswer(
                supported=True, kind="window",
                answer=f"No matching recorded events {window}.",
                scope_note=_SCOPE_NOTE,
            )
        return HistoryAnswer(
            supported=True, kind="window",
            answer=f"{len(events)} recorded event(s) {window}.",
            events=events, scope_note=_SCOPE_NOTE,
        )

    return HistoryAnswer(
        supported=False, kind="unsupported",
        answer=(
            "That question is not supported yet. This prototype answers "
            "questions about recorded events, their timing, and action "
            "delivery."
        ),
        suggestions=SUPPORTED_EXAMPLES,
    )
