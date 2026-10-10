"""Live acceptance run against a running backend + receiver.

This is the guide's step-11 check, automated. Unlike the unit tests it talks to
the real HTTP API, the real database and a real webhook receiver, and it waits
on wall-clock time exactly as a judge would.

    # terminal 1
    python -m uvicorn app.receiver:app --host 127.0.0.1 --port 9000
    # terminal 2 (from backend/)
    python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
    # terminal 3
    python scripts/acceptance.py

Results are written to docs/acceptance-results.md. A 20-case run is a demo
check, not proof of production accuracy, and the report says so.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib import error, request

API = "http://127.0.0.1:8000"
RECEIVER = "http://127.0.0.1:9000"
REPO_ROOT = Path(__file__).resolve().parents[1]

PASS, FAIL = "PASS", "FAIL"


def call(url: str, method: str = "GET", body: dict | None = None, timeout: float = 20.0):
    data = json.dumps(body).encode() if body is not None else None
    req = request.Request(url, data=data, method=method)
    if data:
        req.add_header("Content-Type", "application/json")
    # The local servers must not go through any configured HTTP proxy.
    opener = request.build_opener(request.ProxyHandler({}))
    try:
        with opener.open(req, timeout=timeout) as resp:
            raw = resp.read().decode()
            return json.loads(raw) if raw else None
    except error.HTTPError as exc:
        return {"_http_error": exc.code, "_body": exc.read().decode()[:400]}


class Report:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str]] = []

    def check(self, scenario: str, expected: str, ok: bool, detail: str = "") -> bool:
        self.rows.append((scenario, expected, f"{PASS if ok else FAIL} {detail}".strip()))
        marker = "ok  " if ok else "FAIL"
        print(f"  [{marker}] {scenario} :: {detail}")
        return ok

    @property
    def failures(self) -> int:
        return sum(1 for _, _, result in self.rows if result.startswith(FAIL))

    def write(self, path: Path, meta: dict) -> None:
        lines = [
            "# Acceptance results",
            "",
            f"Run at {datetime.now(timezone.utc).isoformat()}",
            "",
            "| setting | value |",
            "|---|---|",
        ]
        lines += [f"| {k} | {v} |" for k, v in meta.items()]
        lines += [
            "",
            "| scenario | expected | result |",
            "|---|---|---|",
        ]
        lines += [f"| {s} | {e} | {r} |" for s, e, r in self.rows]
        lines += [
            "",
            f"**{len(self.rows) - self.failures}/{len(self.rows)} checks passed.**",
            "",
            "> These are demo gates, not production accuracy claims. The run uses "
            "a synthetic scene and the heuristic vision provider unless "
            "`GEMINI_API_KEY` is set; swap in real footage and a real model "
            "before quoting any number from this table.",
            "",
        ]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines))
        print(f"\nwrote {path}")


def wait_for(predicate, timeout: float, interval: float = 1.0, label: str = ""):
    """Poll until predicate returns a truthy value or the timeout expires."""
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        last = predicate()
        if last:
            return last
        time.sleep(interval)
    if label:
        print(f"    (timed out after {timeout:.0f}s waiting for {label})")
    return last


def events_for(workflow_id: str) -> list[dict]:
    result = call(f"{API}/api/events?workflow_id={workflow_id}")
    return result if isinstance(result, list) else []


def wait_for_terminal_action(workflow_id: str, timeout: float = 150.0):
    """Wait until an event exists AND its action has a terminal status.

    An action is committed as ``pending`` in the same transaction as its event,
    and only reaches succeeded/failed once the dispatcher finishes its bounded
    retries. Reading it the instant the event appears is a race.
    """
    def probe():
        for event in events_for(workflow_id):
            for action in event.get("actions", []):
                if action["status"] in ("succeeded", "failed"):
                    return event, action
        return None

    return wait_for(probe, timeout=timeout, label=f"terminal action on {workflow_id}")


def make_workflow(wf_id: str, action: dict, duration: float = 4.0,
                  min_samples: int = 2) -> dict:
    return {
        "workflow": {
            "id": wf_id, "version": 1, "camera_id": "cam_01",
            "condition_type": "region_empty",
            "target_description": "supply tray contains no bottles",
            "roi": {"x": 0.1, "y": 0.2, "w": 0.5, "h": 0.4},
            "duration_seconds": duration, "min_samples": min_samples,
            "max_gap_seconds": 5.0, "clear_samples": 3,
            "action": action, "enabled": True,
        }
    }


def main() -> int:
    report = Report()
    health = call(f"{API}/api/health")
    if not health or health.get("_http_error"):
        print("backend is not reachable at", API)
        return 2

    print(f"backend ok | provider={health['vision_provider']} "
          f"| real_model={health['using_real_model']}\n")
    call(f"{RECEIVER}/control/reset", "POST")

    # ---------------------------------------------------------------
    print("Scenario 1: condition longer than duration -> one event + one task")
    suffix = str(int(time.time()))[-6:]
    wf_task = f"wf_acc_task_{suffix}"
    call(f"{API}/api/workflows", "POST", make_workflow(wf_task, {
        "type": "create_task", "title": "Restock the supply tray"}))

    found = wait_for(lambda: events_for(wf_task), timeout=90,
                     label="first event")
    report.check("Condition lasts longer than configured duration",
                 "One event and one task with correct evidence",
                 bool(found) and len(found) == 1,
                 f"{len(found or [])} event(s)")

    if found:
        event = found[0]
        snap = event.get("snapshot_path")
        report.check("Every event has evidence", "Snapshot file exists",
                     bool(snap) and Path(snap).is_file(),
                     Path(snap).name if snap else "none")

        settled_task = wait_for_terminal_action(wf_task, timeout=30)
        succeeded = []
        if settled_task:
            _, action = settled_task
            succeeded = [action] if action["status"] == "succeeded" else []
        receipt = succeeded[0]["receipt"] if succeeded else ""
        report.check("Action receipt recorded",
                     "create_task returns a task id",
                     bool(succeeded) and str(receipt).startswith("task:"),
                     str(receipt)[:40])

        tasks = call(f"{API}/api/tasks") or []
        mine = [t for t in tasks if t["event_id"] == event["id"]]
        report.check("Zero duplicate internal tasks",
                     "Exactly one task for the event",
                     len(mine) == 1, f"{len(mine)} task(s)")

    # ---------------------------------------------------------------
    print("\nScenario 2: condition persists -> no duplicate tasks")
    time.sleep(12)
    again = events_for(wf_task)
    report.check("Condition persists for several minutes",
                 "No duplicate tasks during the same episode",
                 len(again) == len(found or []),
                 f"{len(again)} event(s) after 12 more seconds")

    # ---------------------------------------------------------------
    print("\nScenario 3: lens covered -> unknown, no false confirmation")
    status = call(f"{API}/api/workflows/{wf_task}/status") or {}
    obs_seen = wait_for(
        lambda: (call(f"{API}/api/workflows/{wf_task}/status") or {})
        .get("latest_observation", {}).get("result") == "unknown",
        timeout=70, label="an unknown observation (covered lens phase)")
    report.check("Lens covered or subject occluded",
                 "Unknown; no false confirmation",
                 bool(obs_seen),
                 "observed unknown" if obs_seen else "phase not reached in window")

    # ---------------------------------------------------------------
    print("\nScenario 4: webhook fails, then recovers on retry")
    wf_hook = f"wf_acc_hook_{suffix}"
    call(f"{RECEIVER}/control/fail?enabled=true&status=503", "POST")
    call(f"{API}/api/workflows", "POST", make_workflow(wf_hook, {
        "type": "webhook", "destination_id": "test_receiver",
        "title": "Tray empty"}))

    settled = wait_for_terminal_action(wf_hook, timeout=150)
    failed_action = None
    if settled:
        _, candidate = settled
        if candidate["status"] == "failed":
            failed_action = candidate
    report.check("Webhook receiver fails",
                 "Failure is visible, not swallowed",
                 failed_action is not None,
                 (failed_action or {}).get("last_error", "")[:45] or "no failure recorded")

    if failed_action:
        call(f"{RECEIVER}/control/fail?enabled=false", "POST")
        retried = call(f"{API}/api/actions/{failed_action['id']}/retry", "POST")
        report.check("Receiver recovers, retry succeeds",
                     "Action status becomes succeeded",
                     bool(retried) and retried.get("status") == "succeeded",
                     (retried or {}).get("status", "?"))

        # Deliver the same action again; the receiver must deduplicate.
        call(f"{API}/api/actions/{failed_action['id']}/retry", "POST")
        records = call(f"{RECEIVER}/records") or {}
        report.check("Receiver deduplicates",
                     "Repeated delivery creates no second record",
                     records.get("unique_records", 0) == 1
                     and records.get("total_deliveries", 0) > 1,
                     f"unique={records.get('unique_records')} "
                     f"deliveries={records.get('total_deliveries')} "
                     f"suppressed={records.get('duplicates_suppressed')}")

    # ---------------------------------------------------------------
    print("\nScenario 5: edit while running -> old results cannot trigger")
    patched = call(f"{API}/api/workflows/{wf_task}", "PATCH",
                   {"duration_seconds": 9.0})
    report.check("Edit workflow while inference is running",
                 "Version increments so old results are rejected",
                 bool(patched) and patched.get("version") == 2
                 and patched.get("version_bumped") is True,
                 f"version={(patched or {}).get('version')}")

    # ---------------------------------------------------------------
    print("\nScenario 6: history answers are deterministic and scoped")
    answer = call(f"{API}/api/history/query", "POST",
                  {"question": "How many events occurred today?",
                   "workflow_id": wf_task})
    report.check("Counts are deterministic",
                 "Count answered from the database",
                 bool(answer) and answer.get("supported") is True
                 and "recorded" in answer.get("answer", ""),
                 (answer or {}).get("answer", "")[:40])

    unknown_q = call(f"{API}/api/history/query", "POST",
                     {"question": "Who walked past the camera?"})
    report.check("Unsupported question is explained",
                 "Explains instead of inventing capability",
                 bool(unknown_q) and unknown_q.get("supported") is False,
                 (unknown_q or {}).get("answer", "")[:45])

    empty_q = call(f"{API}/api/history/query", "POST",
                   {"question": "When did this rule last trigger?",
                    "workflow_id": "wf_does_not_exist"})
    report.check("No data is not 'nothing happened'",
                 "Says no matching recorded events",
                 bool(empty_q) and "no matching recorded events"
                 in empty_q.get("answer", "").lower(),
                 (empty_q or {}).get("answer", "")[:45])

    # ---------------------------------------------------------------
    # Clean up so the demo history starts empty.
    call(f"{API}/api/workflows/{wf_task}", "PATCH", {"enabled": False})
    call(f"{API}/api/workflows/{wf_hook}", "PATCH", {"enabled": False})

    report.write(REPO_ROOT / "docs" / "acceptance-results.md", {
        "vision provider": health["vision_provider"],
        "vision model": health["vision_model"],
        "using real model": health["using_real_model"],
        "camera source": (call(f"{API}/api/cameras") or [{}])[0].get("source_kind", "?"),
        "sample interval (s)": health["sample_interval_seconds"],
    })

    print(f"\n{len(report.rows) - report.failures}/{len(report.rows)} checks passed")
    return 1 if report.failures else 0


if __name__ == "__main__":
    sys.exit(main())
