"""SQLite storage.

Owner: C. Every write goes through one serialized service (guide step 8): WAL
mode plus a busy timeout plus a process-wide lock. SQLite tolerates concurrent
readers happily and concurrent writers badly, and the capture worker, the
inference worker and the HTTP handlers all write.

Evidence lives on disk; the database stores paths, never blobs.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

from .schemas import (
    ActionRecord,
    ActionStatus,
    ActionType,
    Episode,
    Event,
    EventStatus,
    Observation,
    Task,
    Workflow,
    utcnow,
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS cameras (
    id            TEXT PRIMARY KEY,
    source_kind   TEXT NOT NULL,
    source_ref    TEXT NOT NULL,
    label         TEXT NOT NULL DEFAULT '',
    last_frame_at TEXT,
    health        TEXT NOT NULL DEFAULT 'disconnected'
);

CREATE TABLE IF NOT EXISTS workflows (
    id         TEXT NOT NULL,
    version    INTEGER NOT NULL,
    payload    TEXT NOT NULL,
    enabled    INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    PRIMARY KEY (id, version)
);

CREATE TABLE IF NOT EXISTS observations (
    id               TEXT PRIMARY KEY,
    workflow_id      TEXT NOT NULL,
    workflow_version INTEGER NOT NULL,
    camera_id        TEXT NOT NULL,
    frame_id         INTEGER NOT NULL,
    captured_at_utc  TEXT NOT NULL,
    processed_at_utc TEXT NOT NULL,
    result           TEXT NOT NULL,
    explanation      TEXT NOT NULL DEFAULT '',
    model_id         TEXT NOT NULL DEFAULT '',
    inference_ms     INTEGER NOT NULL DEFAULT 0,
    evidence_path    TEXT
);
CREATE INDEX IF NOT EXISTS idx_obs_workflow_time
    ON observations (workflow_id, captured_at_utc DESC);

CREATE TABLE IF NOT EXISTS episodes (
    id          TEXT PRIMARY KEY,
    workflow_id TEXT NOT NULL,
    payload     TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS events (
    id               TEXT PRIMARY KEY,
    workflow_id      TEXT NOT NULL,
    workflow_version INTEGER NOT NULL DEFAULT 1,
    episode_id       TEXT NOT NULL,
    first_positive_at TEXT NOT NULL,
    triggered_at     TEXT NOT NULL,
    status           TEXT NOT NULL,
    snapshot_path    TEXT,
    clip_path        TEXT,
    explanation      TEXT NOT NULL DEFAULT '',
    -- One episode yields at most one event. This constraint, not dispatcher
    -- discipline, is what makes duplicate prevention survive a crash.
    UNIQUE (workflow_id, episode_id)
);
CREATE INDEX IF NOT EXISTS idx_events_time ON events (triggered_at DESC);

CREATE TABLE IF NOT EXISTS actions (
    id              TEXT PRIMARY KEY,
    event_id        TEXT NOT NULL,
    type            TEXT NOT NULL,
    status          TEXT NOT NULL,
    attempts        INTEGER NOT NULL DEFAULT 0,
    last_error      TEXT,
    receipt         TEXT,
    idempotency_key TEXT NOT NULL DEFAULT '',
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    -- At most one action of each type per event.
    UNIQUE (event_id, type),
    FOREIGN KEY (event_id) REFERENCES events (id)
);
CREATE INDEX IF NOT EXISTS idx_actions_pending ON actions (status, created_at);

CREATE TABLE IF NOT EXISTS tasks (
    id            TEXT PRIMARY KEY,
    title         TEXT NOT NULL,
    workflow_id   TEXT NOT NULL,
    event_id      TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    completed     INTEGER NOT NULL DEFAULT 0,
    evidence_path TEXT,
    UNIQUE (event_id)
);
"""


def _iso(value: Optional[datetime]) -> Optional[str]:
    return value.isoformat() if value else None


class Database:
    """Serialized SQLite access.

    ``check_same_thread=False`` plus an explicit lock gives one connection that
    every worker can use, with writes strictly ordered.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            str(self.path), check_same_thread=False, timeout=30.0
        )
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            # WAL lets the UI poll while the capture loop writes.
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA busy_timeout=30000")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    @contextmanager
    def write(self) -> Iterator[sqlite3.Connection]:
        """One serialized, atomic write transaction."""
        with self._lock:
            try:
                yield self._conn
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    def query(self, sql: str, params: Iterable[Any] = ()) -> list[sqlite3.Row]:
        with self._lock:
            return list(self._conn.execute(sql, tuple(params)).fetchall())

    # -- workflows ------------------------------------------------------

    def save_workflow(self, wf: Workflow) -> None:
        with self.write() as c:
            c.execute(
                "INSERT OR REPLACE INTO workflows (id, version, payload, enabled, created_at)"
                " VALUES (?,?,?,?,?)",
                (
                    wf.id,
                    wf.version,
                    wf.model_dump_json(),
                    int(wf.enabled),
                    utcnow().isoformat(),
                ),
            )

    def get_workflow(self, workflow_id: str, version: Optional[int] = None) -> Optional[Workflow]:
        if version is None:
            rows = self.query(
                "SELECT payload FROM workflows WHERE id=? ORDER BY version DESC LIMIT 1",
                (workflow_id,),
            )
        else:
            rows = self.query(
                "SELECT payload FROM workflows WHERE id=? AND version=?",
                (workflow_id, version),
            )
        return Workflow.model_validate_json(rows[0]["payload"]) if rows else None

    def list_workflows(self) -> list[Workflow]:
        rows = self.query(
            "SELECT payload FROM workflows w WHERE version ="
            " (SELECT MAX(version) FROM workflows x WHERE x.id = w.id)"
        )
        return [Workflow.model_validate_json(r["payload"]) for r in rows]

    # -- observations ---------------------------------------------------

    def save_observation(self, obs: Observation) -> None:
        with self.write() as c:
            c.execute(
                "INSERT OR IGNORE INTO observations (id, workflow_id, workflow_version,"
                " camera_id, frame_id, captured_at_utc, processed_at_utc, result,"
                " explanation, model_id, inference_ms, evidence_path)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    obs.id,
                    obs.workflow_id,
                    obs.workflow_version,
                    obs.camera_id,
                    obs.frame_id,
                    obs.captured_at_utc.isoformat(),
                    obs.processed_at_utc.isoformat(),
                    obs.result.value,
                    obs.explanation,
                    obs.model_id,
                    obs.inference_ms,
                    obs.evidence_path,
                ),
            )

    def latest_observation(self, workflow_id: str) -> Optional[dict[str, Any]]:
        rows = self.query(
            "SELECT * FROM observations WHERE workflow_id=?"
            " ORDER BY captured_at_utc DESC LIMIT 1",
            (workflow_id,),
        )
        return dict(rows[0]) if rows else None

    # -- episodes -------------------------------------------------------

    def save_episode(self, ep: Episode) -> None:
        with self.write() as c:
            c.execute(
                "INSERT OR REPLACE INTO episodes (id, workflow_id, payload, updated_at)"
                " VALUES (?,?,?,?)",
                (ep.id, ep.workflow_id, ep.model_dump_json(), utcnow().isoformat()),
            )

    def load_episodes(self, workflow_id: str) -> list[Episode]:
        rows = self.query(
            "SELECT payload FROM episodes WHERE workflow_id=? ORDER BY updated_at DESC",
            (workflow_id,),
        )
        return [Episode.model_validate_json(r["payload"]) for r in rows]

    # -- events + actions (atomic) --------------------------------------

    def commit_event_with_action(
        self,
        event: Event,
        action: ActionRecord,
        task: Optional[Task] = None,
    ) -> bool:
        """Insert the event and its pending action in one transaction.

        Returns False when the episode already produced an event, which is the
        normal outcome of a retry after a crash -- not an error.

        Committing both together is what prevents an event existing with no
        scheduled action after a crash between two separate writes.
        """
        with self.write() as c:
            existing = c.execute(
                "SELECT id FROM events WHERE workflow_id=? AND episode_id=?",
                (event.workflow_id, event.episode_id),
            ).fetchone()
            if existing:
                return False

            c.execute(
                "INSERT INTO events (id, workflow_id, workflow_version, episode_id,"
                " first_positive_at, triggered_at, status, snapshot_path, clip_path,"
                " explanation) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    event.id,
                    event.workflow_id,
                    event.workflow_version,
                    event.episode_id,
                    event.first_positive_at.isoformat(),
                    event.triggered_at.isoformat(),
                    event.status.value,
                    event.snapshot_path,
                    event.clip_path,
                    event.explanation,
                ),
            )
            c.execute(
                "INSERT INTO actions (id, event_id, type, status, attempts, last_error,"
                " receipt, idempotency_key, created_at, updated_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    action.id,
                    action.event_id,
                    action.type.value,
                    action.status.value,
                    action.attempts,
                    action.last_error,
                    action.receipt,
                    action.idempotency_key,
                    action.created_at.isoformat(),
                    action.updated_at.isoformat(),
                ),
            )
            if task is not None:
                c.execute(
                    "INSERT OR IGNORE INTO tasks (id, title, workflow_id, event_id,"
                    " created_at, completed, evidence_path) VALUES (?,?,?,?,?,?,?)",
                    (
                        task.id,
                        task.title,
                        task.workflow_id,
                        task.event_id,
                        task.created_at.isoformat(),
                        int(task.completed),
                        task.evidence_path,
                    ),
                )
            return True

    def attach_evidence(
        self,
        event_id: str,
        snapshot_path: Optional[str] = None,
        clip_path: Optional[str] = None,
    ) -> None:
        sets, params = [], []
        if snapshot_path is not None:
            sets.append("snapshot_path=?")
            params.append(snapshot_path)
        if clip_path is not None:
            sets.append("clip_path=?")
            params.append(clip_path)
        if not sets:
            return
        params.append(event_id)
        with self.write() as c:
            c.execute(f"UPDATE events SET {', '.join(sets)} WHERE id=?", params)

    def set_event_status(self, event_id: str, status: EventStatus) -> None:
        with self.write() as c:
            c.execute("UPDATE events SET status=? WHERE id=?", (status.value, event_id))

    def close_episode_event(self, workflow_id: str, episode_id: str) -> None:
        with self.write() as c:
            c.execute(
                "UPDATE events SET status=? WHERE workflow_id=? AND episode_id=?",
                (EventStatus.CLOSED.value, workflow_id, episode_id),
            )

    def get_event(self, event_id: str) -> Optional[dict[str, Any]]:
        rows = self.query("SELECT * FROM events WHERE id=?", (event_id,))
        return dict(rows[0]) if rows else None

    def list_events(
        self,
        workflow_id: Optional[str] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Parameterized filter. The history endpoint compiles questions into
        these arguments; it never builds SQL from model output."""
        sql = "SELECT * FROM events WHERE 1=1"
        params: list[Any] = []
        if workflow_id:
            sql += " AND workflow_id=?"
            params.append(workflow_id)
        if since:
            sql += " AND triggered_at>=?"
            params.append(since.isoformat())
        if until:
            sql += " AND triggered_at<=?"
            params.append(until.isoformat())
        sql += " ORDER BY triggered_at DESC LIMIT ?"
        params.append(max(1, min(limit, 1000)))
        return [dict(r) for r in self.query(sql, params)]

    # -- actions / tasks -------------------------------------------------

    def pending_actions(self, limit: int = 20) -> list[ActionRecord]:
        rows = self.query(
            "SELECT * FROM actions WHERE status=? ORDER BY created_at LIMIT ?",
            (ActionStatus.PENDING.value, limit),
        )
        return [self._row_to_action(r) for r in rows]

    def get_action(self, action_id: str) -> Optional[ActionRecord]:
        rows = self.query("SELECT * FROM actions WHERE id=?", (action_id,))
        return self._row_to_action(rows[0]) if rows else None

    def list_actions(self, event_id: Optional[str] = None) -> list[ActionRecord]:
        if event_id:
            rows = self.query("SELECT * FROM actions WHERE event_id=?", (event_id,))
        else:
            rows = self.query("SELECT * FROM actions ORDER BY created_at DESC LIMIT 200")
        return [self._row_to_action(r) for r in rows]

    @staticmethod
    def _row_to_action(r: sqlite3.Row) -> ActionRecord:
        return ActionRecord(
            id=r["id"],
            event_id=r["event_id"],
            type=ActionType(r["type"]),
            status=ActionStatus(r["status"]),
            attempts=r["attempts"],
            last_error=r["last_error"],
            receipt=r["receipt"],
            idempotency_key=r["idempotency_key"],
            created_at=datetime.fromisoformat(r["created_at"]),
            updated_at=datetime.fromisoformat(r["updated_at"]),
        )

    def update_action(self, action: ActionRecord) -> None:
        with self.write() as c:
            c.execute(
                "UPDATE actions SET status=?, attempts=?, last_error=?, receipt=?,"
                " updated_at=? WHERE id=?",
                (
                    action.status.value,
                    action.attempts,
                    action.last_error,
                    action.receipt,
                    utcnow().isoformat(),
                    action.id,
                ),
            )

    def create_task(self, task: Task) -> bool:
        with self.write() as c:
            cur = c.execute(
                "INSERT OR IGNORE INTO tasks (id, title, workflow_id, event_id,"
                " created_at, completed, evidence_path) VALUES (?,?,?,?,?,?,?)",
                (
                    task.id,
                    task.title,
                    task.workflow_id,
                    task.event_id,
                    task.created_at.isoformat(),
                    int(task.completed),
                    task.evidence_path,
                ),
            )
            return cur.rowcount > 0

    def list_tasks(self, completed: Optional[bool] = None) -> list[dict[str, Any]]:
        if completed is None:
            rows = self.query("SELECT * FROM tasks ORDER BY created_at DESC LIMIT 200")
        else:
            rows = self.query(
                "SELECT * FROM tasks WHERE completed=? ORDER BY created_at DESC LIMIT 200",
                (int(completed),),
            )
        return [dict(r) for r in rows]

    def count_events(
        self, workflow_id: Optional[str] = None, since: Optional[datetime] = None
    ) -> int:
        sql = "SELECT COUNT(*) AS n FROM events WHERE 1=1"
        params: list[Any] = []
        if workflow_id:
            sql += " AND workflow_id=?"
            params.append(workflow_id)
        if since:
            sql += " AND triggered_at>=?"
            params.append(since.isoformat())
        return int(self.query(sql, params)[0]["n"])
