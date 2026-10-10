# Acceptance results

Run at 2026-10-10T02:31:22.318150+00:00

| setting | value |
|---|---|
| vision provider | heuristic |
| vision model | heuristic-v1 |
| using real model | False |
| camera source | synthetic |
| sample interval (s) | 1.0 |

| scenario | expected | result |
|---|---|---|
| Condition lasts longer than configured duration | One event and one task with correct evidence | PASS 1 event(s) |
| Every event has evidence | Snapshot file exists | PASS evt_bad2a66fbde7.jpg |
| Action receipt recorded | create_task returns a task id | PASS task:task_e458da21ad7c |
| Zero duplicate internal tasks | Exactly one task for the event | PASS 1 task(s) |
| Condition persists for several minutes | No duplicate tasks during the same episode | PASS 1 event(s) after 12 more seconds |
| Lens covered or subject occluded | Unknown; no false confirmation | PASS observed unknown |
| Webhook receiver fails | Failure is visible, not swallowed | PASS HTTP 503: {"error":"simulated outage"} |
| Receiver recovers, retry succeeds | Action status becomes succeeded | PASS succeeded |
| Receiver deduplicates | Repeated delivery creates no second record | PASS unique=1 deliveries=4 suppressed=3 |
| Edit workflow while inference is running | Version increments so old results are rejected | PASS version=2 |
| Counts are deterministic | Count answered from the database | PASS 2 recorded events today. |
| Unsupported question is explained | Explains instead of inventing capability | PASS That question is not supported yet. This prot |
| No data is not 'nothing happened' | Says no matching recorded events | PASS No matching recorded events. |

**13/13 checks passed.**

> These are demo gates, not production accuracy claims. The run uses a synthetic scene and the heuristic vision provider unless `GEMINI_API_KEY` is set; swap in real footage and a real model before quoting any number from this table.
