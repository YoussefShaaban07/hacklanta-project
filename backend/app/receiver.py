"""A webhook receiver you control, for demos and acceptance testing.

Run it on its own port:

    python -m uvicorn app.receiver:app --host 127.0.0.1 --port 9000

It deduplicates on ``Idempotency-Key``, because the sender is at-least-once.
That is not a workaround -- it is what any correct receiver must do, and
having it here makes the guarantee visible to anyone reading the demo.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse

log = logging.getLogger(__name__)

app = FastAPI(title="test webhook receiver", version="1.0.0")

_lock = threading.Lock()
#: idempotency key -> the record created for it
_records: dict[str, dict[str, Any]] = {}
_deliveries: list[dict[str, Any]] = []
#: Set via /control/fail to simulate an outage during acceptance testing.
_fail_mode = {"enabled": False, "status": 503}


@app.post("/hook")
async def hook(request: Request, idempotency_key: str = Header(default="")) -> JSONResponse:
    body = await request.json()
    key = idempotency_key or body.get("idempotency_key") or ""
    now = datetime.now(timezone.utc).isoformat()

    with _lock:
        _deliveries.append({"received_at": now, "key": key, "body": body})
        if _fail_mode["enabled"]:
            log.warning("simulated failure for key %s", key)
            return JSONResponse(
                {"error": "simulated outage"}, status_code=_fail_mode["status"]
            )
        if key and key in _records:
            # Already acted on. Acknowledge without creating a second record.
            return JSONResponse(
                {"deduplicated": True, "record_id": _records[key]["record_id"]},
                status_code=200,
            )
        record = {
            "record_id": f"rec_{len(_records) + 1:04d}",
            "created_at": now,
            "event_id": body.get("event_id"),
            "workflow_id": body.get("workflow_id"),
            "explanation": body.get("explanation"),
        }
        if key:
            _records[key] = record

    log.info("accepted %s for event %s", record["record_id"], record["event_id"])
    return JSONResponse({"ok": True, **record}, status_code=201)


@app.get("/records")
def records() -> dict[str, Any]:
    """What the receiver actually created -- the demo's proof of delivery."""
    with _lock:
        return {
            "records": list(_records.values()),
            "unique_records": len(_records),
            "total_deliveries": len(_deliveries),
            "duplicates_suppressed": len(_deliveries) - len(_records),
        }


@app.post("/control/fail")
def set_fail(enabled: bool = True, status: int = 503) -> dict[str, Any]:
    with _lock:
        _fail_mode["enabled"] = enabled
        _fail_mode["status"] = status
    return dict(_fail_mode)


@app.post("/control/reset")
def reset() -> dict[str, str]:
    with _lock:
        _records.clear()
        _deliveries.clear()
        _fail_mode["enabled"] = False
    return {"status": "reset"}
